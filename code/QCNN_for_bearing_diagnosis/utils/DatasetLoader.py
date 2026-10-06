# # # -*- coding: utf-8 -*-
# """
# DatasetLoader_v3.py
# Author: Ali Naderi - 2025
# Production-ready PyTorch Dataset loader for CWRU data produced by
# Preprocess_refactored_v3_production.py.

# Features:
#  - Supports modes: 'train', 'val', 'test' with optional SNR selection for test.
#  - Memory Efficiency: Supports lazy loading (mmap) for large datasets.
#  - Instance Normalization: Optional per-channel standardization (zero mean, unit var).
#  - Artifact Loading: Auto-loads feature scalers and signal stats for denormalization.
#  - Provenance: Can trace back which original file a sample came from.
# """

# from __future__ import annotations
# import os
# import json
# import logging
# from typing import Optional, Callable, Tuple, Any, Dict, List, Union

# import numpy as np
# import joblib
# import torch
# from torch.utils.data import Dataset

# # ------------------- Logger Setup -------------------
# logger = logging.getLogger("RobustBearingDataset_v3")
# if not logger.handlers:
#     ch = logging.StreamHandler()
#     ch.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(message)s", datefmt="%H:%M:%S"))
#     logger.addHandler(ch)
# logger.setLevel(logging.INFO)


# class RobustBearingDataset(Dataset):
#     def __init__(
#         self,
#         data_path: str,
#         meta_path: Optional[str] = None,
#         mode: str = "train",
#         snr: Optional[int] = None,
#         load_into_memory: bool = True,
#         instance_norm: bool = True, 
#         transform: Optional[Callable[[np.ndarray], np.ndarray]] = None,
#         feat_transform: Optional[Callable[[np.ndarray], np.ndarray]] = None,
#         target_transform: Optional[Callable[[int], int]] = None,
#         load_scaler: bool = True,
#     ):
#         """
#         Args:
#             data_path (str): Path to 'processed_data_robust_v3.npz'.
#             meta_path (str, optional): Path to 'meta.json'. Defaults to same dir as data_path.
#             mode (str): 'train', 'val', or 'test'.
#             snr (int, optional): If mode='test', specifies SNR level (e.g., -6, 0, 6). If None, uses clean test set.
#             load_into_memory (bool): If True, copies data to RAM. If False, uses memory mapping (slower but RAM efficient).
#             instance_norm (bool): If True, applies per-sample, per-channel standardization (mean=0, std=1).
#             transform (callable, optional): Transform applied to signal.
#             feat_transform (callable, optional): Transform applied to features.
#             target_transform (callable, optional): Transform applied to label.
#             load_scaler (bool): Whether to try loading 'feature_scaler.joblib' for inverse feature transform.
#         """
#         self.data_path = data_path
#         self.mode = mode.lower()
#         self.snr = snr
#         self.load_into_memory = load_into_memory
#         self.instance_norm = instance_norm
#         self.transform = transform
#         self.feat_transform = feat_transform
#         self.target_transform = target_transform
#         self.load_scaler_flag = load_scaler

#         if not os.path.exists(self.data_path):
#             raise FileNotFoundError(f"Dataset file not found: {self.data_path}")

#         self.base_dir = os.path.dirname(self.data_path)
        
#         # 1. Load Metadata
#         if meta_path is None:
#             self.meta_path = os.path.join(self.base_dir, "meta.json")
#         else:
#             self.meta_path = meta_path
        
#         self.meta = {}
#         if os.path.exists(self.meta_path):
#             try:
#                 with open(self.meta_path, 'r', encoding='utf-8') as f:
#                     self.meta = json.load(f)
#             except Exception as e:
#                 logger.warning(f"Failed to load meta.json: {e}")
        
#         # 2. Open NPZ Archive
#         try:
#             if self.load_into_memory:
#                 self._archive = np.load(self.data_path, allow_pickle=True)
#             else:
#                 # mmap_mode='r' allows accessing data on disk without loading everything to RAM
#                 self._archive = np.load(self.data_path, allow_pickle=True, mmap_mode='r')
#         except Exception as e:
#             raise RuntimeError(f"Failed to open npz archive '{self.data_path}': {e}")

#         # 3. Determine Keys based on Mode
#         self.sig_key, self.feat_key, self.lbl_key = self._choose_keys()
        
#         # 4. Validate Keys
#         self._validate_keys()

#         # 5. Load Data Pointers
#         if self.load_into_memory:
#             self.signals = np.array(self._archive[self.sig_key])
#             self.features = np.array(self._archive[self.feat_key])
#             self.labels = np.array(self._archive[self.lbl_key])
#             logger.info(f"Loaded {self.mode} set into RAM.")
#         else:
#             self.signals = self._archive[self.sig_key]
#             self.features = self._archive[self.feat_key]
#             self.labels = self._archive[self.lbl_key]
#             logger.info(f"Initialized {self.mode} set (Memory Mapped).")

#         # 6. Load Artifacts (Scaler & Stats)
#         self.scaler = None
#         self.signal_stats = None
        
#         if self.load_scaler_flag:
#             scaler_fp = os.path.join(self.base_dir, "feature_scaler.joblib")
#             if os.path.exists(scaler_fp):
#                 try:
#                     self.scaler = joblib.load(scaler_fp)
#                 except Exception as e:
#                     logger.warning(f"Could not load feature scaler: {e}")

#         stats_fp = os.path.join(self.base_dir, "signal_norm_stats.npz")
#         if os.path.exists(stats_fp):
#             try:
#                 self.signal_stats = dict(np.load(stats_fp))
#             except Exception as e:
#                 logger.warning(f"Could not load signal stats: {e}")

#         # 7. Map Indices to Class Names
#         self.idx_to_label = self._parse_label_map()
        
#         logger.info(f"Dataset Ready: [{self.mode.upper()}] | Samples: {len(self.labels)} | SNR: {self.snr if self.snr is not None else 'Clean'}")

#     def _choose_keys(self) -> Tuple[str, str, str]:
#         """Selects the correct NPZ keys based on mode and SNR."""
#         if self.mode == "train":
#             return "X_train_sig", "X_train_feat", "y_train"
        
#         elif self.mode in ["val", "validation"]:
#             return "X_val_sig", "X_val_feat", "y_val"
        
#         elif self.mode == "test":
#             lbl_key = "y_test"
#             if self.snr is None:
#                 # Clean test set
#                 return "X_test_sig", "X_test_feat", lbl_key
#             else:
#                 # Noisy test set
#                 sig_key = f"test_snr_{self.snr}_sig"
#                 feat_key = f"test_snr_{self.snr}_feat"
#                 return sig_key, feat_key, lbl_key
        
#         else:
#             raise ValueError(f"Invalid mode '{self.mode}'. Use 'train', 'val', or 'test'.")

#     def _validate_keys(self):
#         """Ensures selected keys actually exist in the archive."""
#         archive_keys = self._archive.files if hasattr(self._archive, 'files') else self._archive.keys()
#         missing = []
#         if self.sig_key not in archive_keys: missing.append(self.sig_key)
#         if self.feat_key not in archive_keys: missing.append(self.feat_key)
#         if self.lbl_key not in archive_keys: missing.append(self.lbl_key)
        
#         if missing:
#             avail_sig = [k for k in archive_keys if 'sig' in k]
#             raise KeyError(f"Missing keys {missing} in NPZ. Available signal keys: {avail_sig}")

#     def _parse_label_map(self) -> Dict[int, str]:
#         """Parses the label map from metadata."""
#         if not self.meta or "idx_to_label" not in self.meta:
#             return {}
        
#         raw_map = self.meta["idx_to_label"]
#         # JSON keys are always strings, convert to int for usage
#         try:
#             return {int(k): v for k, v in raw_map.items()}
#         except:
#             return raw_map

#     def __len__(self) -> int:
#         return len(self.labels)

#     def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
#         if idx < 0:
#             idx += len(self)
        
#         # 1. Load Data (copy=False ensures we don't duplicate if already in RAM)
#         # Shape: (3, L)
#         sig = self.signals[idx].astype(np.float32, copy=False)
#         feat = self.features[idx].astype(np.float32, copy=False)
#         lbl = int(self.labels[idx])

#         # 2. Instance Normalization (Per-Channel Standardization)
#         # This makes the model robust to amplitude variations not related to the fault
#         if self.instance_norm:
#             # axis=1 implies computing mean across time (L), separately for each channel (C)
#             mean = np.mean(sig, axis=1, keepdims=True)
#             std = np.std(sig, axis=1, keepdims=True)
#             std[std < 1e-8] = 1e-8  # Avoid division by zero
#             sig = (sig - mean) / std

#         # 3. Apply Transforms
#         if self.transform:
#             sig = self.transform(sig)
#         if self.feat_transform:
#             feat = self.feat_transform(feat)
#         if self.target_transform:
#             lbl = self.target_transform(lbl)

#         # 4. Convert to Tensor
#         sig_t = torch.from_numpy(sig).float()
#         feat_t = torch.from_numpy(feat).float()
#         lbl_t = torch.tensor(lbl, dtype=torch.long)

#         return sig_t, feat_t, lbl_t

#     # ---------------- Helper Methods ----------------

#     def denorm_signal(self, sig_norm: np.ndarray) -> np.ndarray:
#         """
#         Reverses the global MinMax normalization applied in preprocessing.
#         Useful for visualization or physical analysis.
#         Formula used in preprocess: y = 2 * (x - min) / (max - min) - 1
#         Inverse: x = ((y + 1) / 2) * (max - min) + min
#         """
#         if self.signal_stats is None:
#             raise RuntimeError("Signal stats (x_min, x_max) not loaded.")
        
#         x_min = self.signal_stats['x_min'] # Shape (1, 3, 1)
#         x_max = self.signal_stats['x_max']
        
#         denom = x_max - x_min
#         denom[denom == 0] = 1e-6
        
#         # Handle both single sample (C, L) and batch (B, C, L)
#         is_batch = sig_norm.ndim == 3
#         if not is_batch:
#             sig_norm = sig_norm[np.newaxis, ...] # Add batch dim for broadcasting
            
#         # Broadcasting handles shapes: (B, 3, L) vs (1, 3, 1)
#         original = ((sig_norm + 1.0) / 2.0) * denom + x_min
        
#         return original.astype(np.float32) if is_batch else original[0]

#     def get_provenance(self, idx: int) -> Optional[Dict[str, Any]]:
#         """Returns metadata about which original file generated this sample."""
#         if not self.meta: return None
        
#         map_key = f"{self.mode}_file_map"
#         if self.mode == "test" and self.snr is not None:
#              # SNR test sets usually map back to the 'test' file map conceptually
#              map_key = "test_file_map"

#         file_map = self.meta.get(map_key)
#         if not file_map: return None

#         for entry in file_map:
#             # Check if idx falls within start/end range of a file
#             if entry['start'] <= idx < entry['end']:
#                 return entry
#         return None

#     def get_class_name(self, idx: Union[int, torch.Tensor]) -> str:
#         """Returns the class name for a given label index."""
#         if isinstance(idx, torch.Tensor):
#             idx = int(idx.item())
#         return self.idx_to_label.get(idx, f"Class_{idx}")

# def collate_fn(batch: List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
#     """Custom collate function for DataLoader to stack signals, features, and labels."""
#     sigs = torch.stack([b[0] for b in batch])
#     feats = torch.stack([b[1] for b in batch])
#     labels = torch.stack([b[2] for b in batch])
#     return sigs, feats, labels

# # ---------------- CLI Test Block ----------------
# if __name__ == "__main__":
#     # Update this path to your actual file location
#     TEST_FILE = r"D:\Bearing Project\CWRU\CWRU Case 08\QCNN for bearing diagnosis\data\3HP\processed_data_robust_3hp\processed_data_robust_v3.npz"
    
#     if os.path.exists(TEST_FILE):
#         print(f"Testing DatasetLoader with: {TEST_FILE}")
        
#         # Test 1: Train Loader
#         ds_train = RobustBearingDataset(TEST_FILE, mode="train")
#         s, f, y = ds_train[0]
#         print(f"[Train] Sample 0 - Sig: {s.shape}, Feat: {f.shape}, Label: {y} ({ds_train.get_class_name(y)})")
        
#         # Test 2: Test Loader with SNR
#         try:
#             ds_test_noise = RobustBearingDataset(TEST_FILE, mode="test", snr=-6)
#             s2, f2, y2 = ds_test_noise[0]
#             print(f"[Test SNR=-6] Sample 0 - Sig: {s2.shape}, Feat: {f2.shape}")
#         except KeyError as e:
#             print(f"[Test SNR=-6] Skipped (Key not found): {e}")

#         # Test 3: Provenance
#         prov = ds_train.get_provenance(0)
#         print(f"[Provenance] Sample 0 came from: {prov['file'] if prov else 'Unknown'}")
        
#         # Test 4: Denormalization
#         if ds_train.signal_stats:
#             orig = ds_train.denorm_signal(s.numpy())
#             print(f"[Denorm] Signal range restored to: [{orig.min():.4f}, {orig.max():.4f}]")
#     else:
#         print(f"File not found for testing: {TEST_FILE}")

































# -*- coding: utf-8 -*-
"""
DatasetLoader_v3_pro.py
Author: Ali Naderi - 2025 (Production-Ready & Windows-Safe)
Description:
    Robust PyTorch Dataset loader for Paderborn University (PU) processed NPZ.
    Fully compatible with Preprocess_PU_memmap_ultrapro_final.py output.
    Windows-safe: no special unicode characters in print statements.
"""

from __future__ import annotations
import os
import json
import logging
import argparse
from pathlib import Path
from typing import Optional, Callable, Tuple, Any, Dict, List, Union

import numpy as np
import joblib
import torch
from torch.utils.data import Dataset

# ------------------- Logger Setup -------------------
logger = logging.getLogger("RobustBearingDataset_v3")
logger.setLevel(logging.INFO)
if not logger.handlers:
    ch = logging.StreamHandler()
    ch.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(message)s", datefmt="%H:%M:%S"))
    logger.addHandler(ch)


class RobustBearingDataset(Dataset):
    """
    PyTorch Dataset for PU bearing fault data (10-class severity-aware).

    Args:
        npz_path: Path to processed_*.npz file
        mode: 'train' | 'val' | 'test'
        snr: For test mode only; None = clean, else e.g. -10, -6, etc.
        load_into_memory: If True, loads all data to RAM (faster); else uses mmap
        instance_norm: Per-sample, per-channel z-score normalization
        transform: Optional transform on signal (3, 2048)
        feat_transform: Optional transform on features (feat_dim,)
        target_transform: Optional label transform
        load_scaler: Load feature_scaler.joblib if exists
    """
    def __init__(
        self,
        npz_path: str | Path,
        mode: str = "train",
        snr: Optional[float] = None,
        load_into_memory: bool = True,
        instance_norm: bool = True,
        transform: Optional[Callable[[np.ndarray], np.ndarray]] = None,
        feat_transform: Optional[Callable[[np.ndarray], np.ndarray]] = None,
        target_transform: Optional[Callable[[int], int]] = None,
        load_scaler: bool = True,
    ):
        self.npz_path = Path(npz_path).resolve()
        self.mode = mode.lower()
        self.snr = snr
        self.load_into_memory = load_into_memory
        self.instance_norm = instance_norm
        self.transform = transform
        self.feat_transform = feat_transform
        self.target_transform = target_transform
        self.load_scaler_flag = load_scaler

        if not self.npz_path.is_file():
            raise FileNotFoundError(f"NPZ file not found: {self.npz_path}")

        self.base_dir = self.npz_path.parent

        # Load meta.json
        meta_path = self.base_dir / "meta.json"
        self.meta = {}
        if meta_path.is_file():
            try:
                with open(meta_path, encoding="utf-8") as f:
                    self.meta = json.load(f)
            except Exception as e:
                logger.warning(f"Failed to load meta.json: {e}")

        # Open NPZ
        mmap_mode = None if self.load_into_memory else 'r'
        try:
            self._archive = np.load(self.npz_path, allow_pickle=True, mmap_mode=mmap_mode)
        except Exception as e:
            raise RuntimeError(f"Failed to open NPZ: {e}")

        # Select keys
        self.sig_key, self.feat_key, self.lbl_key = self._choose_keys()
        self._validate_keys()

        # Load data references
        if self.load_into_memory:
            self.signals = np.asarray(self._archive[self.sig_key])
            self.features = np.asarray(self._archive[self.feat_key])
            self.labels   = np.asarray(self._archive[self.lbl_key])
            logger.info(f"Loaded {self.mode.upper()} set into RAM | Samples: {len(self.labels)}")
        else:
            self.signals = self._archive[self.sig_key]
            self.features = self._archive[self.feat_key]
            self.labels   = self._archive[self.lbl_key]
            logger.info(f"Initialized {self.mode.upper()} set (mmap) | Samples: {len(self.labels)}")

        # Load scaler and stats
        self.scaler = None
        self.signal_stats = None

        if self.load_scaler_flag:
            scaler_path = self.base_dir / "feature_scaler.joblib"
            if scaler_path.is_file():
                try:
                    self.scaler = joblib.load(scaler_path)
                    logger.info("Feature scaler loaded")
                except Exception as e:
                    logger.warning(f"Failed to load scaler: {e}")

        stats_path = self.base_dir / "signal_norm_stats.npz"
        if stats_path.is_file():
            try:
                self.signal_stats = dict(np.load(stats_path))
                logger.info("Signal stats loaded")
            except Exception as e:
                logger.warning(f"Failed to load signal stats: {e}")

        self.idx_to_label = self._parse_label_map()

        logger.info(
            f"Dataset ready: [{self.mode.upper()}] | "
            f"Samples: {len(self.labels)} | SNR: {self.snr if self.snr is not None else 'Clean'}"
        )

    def _choose_keys(self) -> Tuple[str, str, str]:
        mode = self.mode
        if mode == "train":
            return "X_train_sig", "X_train_feat", "y_train"
        if mode in ("val", "validation"):
            return "X_val_sig", "X_val_feat", "y_val"
        if mode == "test":
            lbl_key = "y_test"
            if self.snr is None:
                return "X_test_sig", "X_test_feat", lbl_key
            else:
                sig_key = f"test_snr_{self.snr}_sig"
                feat_key = f"test_snr_{self.snr}_feat"
                if sig_key not in self._archive.files:
                    logger.warning(f"SNR {self.snr} not found - falling back to clean test")
                    return "X_test_sig", "X_test_feat", lbl_key
                return sig_key, feat_key, lbl_key
        raise ValueError(f"Invalid mode: {mode}")

    def _validate_keys(self):
        available = set(self._archive.files)
        required = {self.sig_key, self.feat_key, self.lbl_key}
        missing = required - available
        if missing:
            sig_keys = [k for k in available if 'sig' in k]
            logger.error(f"Missing keys: {missing}")
            logger.info(f"Available signal keys: {sig_keys}")
            raise KeyError(f"Missing keys in NPZ: {missing}")

    def _parse_label_map(self) -> Dict[int, str]:
        raw = self.meta.get("classes", {})
        try:
            return {int(k): str(v) for k, v in raw.items()}
        except Exception:
            logger.warning("Could not parse label map")
            return {}

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if idx < 0:
            idx += len(self)

        sig = self.signals[idx].astype(np.float32, copy=False)
        feat = self.features[idx].astype(np.float32, copy=False)
        lbl = int(self.labels[idx])

        if self.instance_norm:
            mean = np.mean(sig, axis=1, keepdims=True)
            std = np.std(sig, axis=1, keepdims=True)
            std = np.maximum(std, 1e-8)
            sig = (sig - mean) / std

        if self.transform:
            sig = self.transform(sig)
        if self.feat_transform:
            feat = self.feat_transform(feat)
        if self.target_transform:
            lbl = self.target_transform(lbl)

        return (
            torch.from_numpy(sig).float(),
            torch.from_numpy(feat).float(),
            torch.tensor(lbl, dtype=torch.long)
        )

    def denorm_signal(self, sig_norm: np.ndarray) -> np.ndarray:
        if self.signal_stats is None:
            raise RuntimeError("signal_norm_stats.npz not loaded.")

        x_min = self.signal_stats['x_min']
        x_max = self.signal_stats['x_max']

        denom = x_max - x_min
        denom = np.maximum(denom, 1e-6)

        is_batch = sig_norm.ndim == 3
        if not is_batch:
            sig_norm = sig_norm[np.newaxis, ...]

        original = ((sig_norm + 1.0) / 2.0) * denom + x_min
        return original.astype(np.float32) if is_batch else original[0]

    def get_provenance(self, idx: int) -> Optional[Dict[str, Any]]:
        if not self.meta:
            return None

        map_key = f"{self.mode}_file_map"
        if self.mode == "test" and self.snr is not None:
            map_key = "test_file_map"

        file_map = self.meta.get(map_key, [])
        for entry in file_map:
            if entry.get('start', 0) <= idx < entry.get('end', 0):
                return entry
        return None

    def get_class_name(self, label: int | torch.Tensor) -> str:
        if isinstance(label, torch.Tensor):
            label = int(label.item())
        return self.idx_to_label.get(label, f"Class {label}")

def collate_fn(batch: List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    sigs = torch.stack([b[0] for b in batch])
    feats = torch.stack([b[1] for b in batch])
    labels = torch.stack([b[2] for b in batch])
    return sigs, feats, labels


# ---------------- CLI Test ----------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test RobustBearingDataset_v3")
    parser.add_argument("--npz", required=True, help="Path to processed_pu_ultrapro_final.npz")
    parser.add_argument("--mode", default="train", choices=["train", "val", "test"])
    parser.add_argument("--snr", type=float, default=None, help="SNR for test mode")
    parser.add_argument("--sample-idx", type=int, default=0, help="Sample index")
    parser.add_argument("--verbose", action="store_true")

    args = parser.parse_args()

    if args.verbose:
        logger.setLevel(logging.DEBUG)

    ds = RobustBearingDataset(
        npz_path=args.npz,
        mode=args.mode,
        snr=args.snr,
        load_into_memory=True,
        instance_norm=True
    )

    print(f"\nDataset Info:")
    print(f"  Mode:       {ds.mode.upper()}")
    print(f"  SNR:        {ds.snr if ds.snr is not None else 'Clean'}")
    print(f"  Samples:    {len(ds)}")
    print(f"  Feat dim:   {ds[0][1].shape[0] if len(ds) > 0 else 'N/A'}")
    print(f"  Classes:    {len(ds.idx_to_label)}")

    idx = min(args.sample_idx, len(ds) - 1)
    sig, feat, lbl = ds[idx]
    print(f"\nSample #{idx}:")
    print(f"  Signal shape:   {tuple(sig.shape)}")
    print(f"  Features shape: {tuple(feat.shape)}")
    print(f"  Label:          {lbl.item()} (Class: {ds.get_class_name(lbl)})")

    prov = ds.get_provenance(idx)
    if prov:
        print(f"  Provenance:     {prov.get('file', 'Unknown')} "
              f"(range {prov.get('start')} - {prov.get('end')})")

    if ds.signal_stats:
        orig = ds.denorm_signal(sig.numpy())
        print(f"  Original range after denorm: [{orig.min():.4f}, {orig.max():.4f}]")