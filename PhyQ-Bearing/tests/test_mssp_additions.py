"""v10.2 additions: simulator, ClassBD, classical Fast-Kurtogram baseline, theory identities."""
import numpy as np
import torch

from phyq.sim.bearing_sim import SimConfig, add_impulsive, add_white, make_dataset, simulate_one


def test_simulator_fault_frequency_in_envelope():
    from scipy.signal import firwin, hilbert, lfilter
    rng = np.random.default_rng(0)
    x, info = simulate_one("outer", rng, SimConfig(), rpm=1797.0, return_info=True)
    h = firwin(127, [2000, 4500], pass_zero=False, fs=12000)
    env = np.abs(hilbert(lfilter(h, 1, x)))
    P = np.abs(np.fft.rfft(env - env.mean())) ** 2
    f = np.fft.rfftfreq(len(env), 1 / 12000)
    band = (f > 20) & (f < 600)
    assert abs(f[band][np.argmax(P[band])] - info["f_fault"]) < 2 * (f[1] - f[0]) or \
        abs(f[band][np.argmax(P[band])] - 2 * info["f_fault"]) < 3 * (f[1] - f[0])


def test_dataset_and_noise_shapes():
    X, y = make_dataset(3, seed=1)
    assert X.shape == (12, 1, 2048) and set(y) == {0, 1, 2, 3}
    rng = np.random.default_rng(0)
    Xn = add_white(X, 0, rng)
    snr = 10 * np.log10((X ** 2).mean() / ((Xn - X) ** 2).mean())
    assert abs(snr) < 0.5
    assert add_impulsive(X, rng).shape == X.shape


def test_filtered_kurtosis_formula():
    from scipy.signal import firwin, lfilter
    rng = np.random.default_rng(0)
    n = 0.3 * rng.standard_normal(400_000) + (rng.random(400_000) < 0.01) * rng.standard_normal(400_000) * 10
    k = lambda v: np.mean((v - v.mean()) ** 4) / np.var(v) ** 2  # noqa: E731
    h = firwin(15, [2400, 4200], pass_zero=False, fs=12000)
    rho = (h ** 4).sum() / ((h ** 2).sum()) ** 2
    pred = 3 + (k(n) - 3) * rho
    assert abs(k(lfilter(h, 1, n)) - pred) / pred < 0.1


def test_product_of_bandpass_outputs_is_squared_envelope():
    # Proposition 2: (w*x)^2 has baseband content at the modulation frequency, w*x has none
    fs, t = 12000, np.arange(4096) / 12000
    a = 1 + 0.8 * np.cos(2 * np.pi * 100 * t)
    x = a * np.cos(2 * np.pi * 3000 * t)
    f = np.fft.rfftfreq(len(t), 1 / fs)
    i100 = np.argmin(abs(f - 100))
    lin = np.abs(np.fft.rfft(x))[i100]
    sq = np.abs(np.fft.rfft(x * x))[i100]
    assert sq > 100 * lin


def test_classbd_forward_backward_physics():
    from phyq.models import build_model
    for ph in (False, True):
        m = build_model({"name": "classbd", "physics_loss": ph}, 2, 5, 1024)
        out = m(torch.randn(4, 2, 1024))
        assert out.shape == (4, 5)
        assert (m.aux_loss is not None) == ph
        loss = out.sum() + (m.aux_loss if ph else 0)
        loss.backward()
        m.eval()
        m(torch.randn(2, 2, 1024))
        assert m.aux_loss is None


def test_kurtogram_selects_resonance_band():
    from phyq.classical.kurtogram import kurtogram_bands, ses_features
    rng = np.random.default_rng(0)
    X = np.stack([simulate_one("outer", rng, SimConfig(), rpm=1797.0) for _ in range(4)])[:, None, :]
    F = ses_features(X, 12000.0, nlevel=4, f_max=600, n_bins=60)
    assert F.shape == (4, 63)
    fc = F[:, 60] * 12000.0 / 2           # normalised centre frequency -> Hz
    assert np.all((fc > 1500) & (fc < 5500))
    assert len(kurtogram_bands(12000.0, 3)) == 1 + 2 + 4 + 8 + 3 + 6
