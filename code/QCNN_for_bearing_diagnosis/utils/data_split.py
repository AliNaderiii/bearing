# from matplotlib import pyplot as plt
# from scipy.io import loadmat, savemat
# import numpy as np
# import os
# from sklearn import preprocessing  # 0-1 encoding
# from sklearn.model_selection import StratifiedShuffleSplit  # Randomly divide to ensure the same proportion of each category

# '''
# Splitting the data set and generating noisy data
# '''

# def wgn(x, snr):
#     snr = 10 ** (snr / 10.0)
#     xpower = np.sum(np.absolute(x) ** 2, axis=0) / x.shape[0]
#     npower = xpower / snr
#     return np.random.standard_normal(x.shape) * np.sqrt(npower)


# def add_noise(data, snr_num):
#     rand_data = wgn(data, snr_num)
#     n_data = data + rand_data
#     return n_data, rand_data

# def preprocess(d_path, noise=False, snr=0):
#     filenames = os.listdir(d_path)

#     def capture():
#         files = {}
#         for i in filenames:
#             file_path = os.path.join(d_path, i)
#             file = loadmat(file_path)
#             file_keys = file.keys()
#             for key in file_keys:
#                 if 'DE' in key:
#                     files[i] = file[key].ravel()
#         return files

#     def slice_enc(data, noise, snr):

#         noised_data_dict1 = {}
#         noise_dict1 = {}
#         noised_data_dict2 = {}
#         noise_dict2 = {}
#         for key, val in data.items():
#             slice_data = val
#             train_data= slice_data[:len(slice_data) * 2 // 3]
#             test_data = slice_data[len(slice_data) * 2 // 3:]
#             if noise:
#                 noised_data1, white_noise1 = add_noise(train_data, snr)
#                 noised_data2, white_noise2 = add_noise(test_data, snr)

#             noised_data_dict1[key] = noised_data1
#             noise_dict1[key] = white_noise1
#             noised_data_dict2[key] = noised_data2
#             noise_dict2[key] = white_noise2
#         return noised_data_dict1, noise_dict1, noised_data_dict2, noise_dict2

#     def save_mat(d_path, noised_data_dict1, noised_data_dict2, snr):
#         path1 = d_path + '_TrainNoised_' + str(snr)
#         if not os.path.exists(path1):
#             os.mkdir(path1)
#         for k, dat in noised_data_dict1.items():
#             savemat(os.path.join(path1, k), {'DE': dat})

#         path1 = d_path + '_TestNoised_' + str(snr)
#         if not os.path.exists(path1):
#             os.mkdir(path1)
#         for k, dat in noised_data_dict2.items():
#             savemat(os.path.join(path1, k), {'DE': dat})

#     # Read a dictionary of data from all .mat files
#     data = capture()
#     noised_data_dict1, noise_dict1, noised_data_dict2, noise_dict2 = slice_enc(data, noise, snr)
#     save_mat(d_path, noised_data_dict1, noised_data_dict2, snr)


# if __name__ == "__main__":
#     path = r"D:\Bearing Project\CWRU\CWRU Case 08\QCNN for bearing diagnosis\data\0HP" # change dataset file folders to split data
#     preprocess(d_path=path,
#                 noise=True,
#                 snr= 0) # We can choice -10, -8, ..., 0, ...,  8, 10



# # data_split_fixed.py
# import os
# from scipy.io import loadmat, savemat
# import numpy as np

# def wgn(x, snr, rng=None):
#     """
#     Generate white Gaussian noise with desired SNR (dB) for signal x.
#     Returns noise array of same shape as x.
#     """
#     if rng is None:
#         rng = np.random.RandomState(None)
#     x = np.asarray(x).astype(np.float64)
#     # ensure 1D vector for power calc per signal
#     # if input is 2D (L,1) -> ravel
#     x_flat = x.ravel()
#     p_signal = np.mean(x_flat ** 2)
#     p_noise = p_signal / (10 ** (snr / 10.0))
#     noise = rng.standard_normal(size=x_flat.shape) * np.sqrt(p_noise)
#     return noise.reshape(x.shape)

# def add_noise(data, snr, rng=None):
#     if rng is None:
#         rng = np.random.RandomState(None)
#     noise = wgn(data, snr, rng=rng)
#     return data + noise, noise

# def capture_de_signals(d_path, min_length=2048):
#     """
#     Read all .mat in d_path and return dict filename->1D array for DE channel (if present).
#     """
#     files = {}
#     for fname in sorted(os.listdir(d_path)):
#         if not fname.endswith('.mat'):
#             continue
#         try:
#             mat = loadmat(os.path.join(d_path, fname))
#         except Exception:
#             continue
#         for key in mat:
#             if key.startswith('__'):
#                 continue
#             if 'de' in key.lower():  # case-insensitive match
#                 arr = np.asarray(mat[key])
#                 # flatten (N,1) or (1,N) -> 1D
#                 arr1d = arr.ravel()
#                 if arr1d.size >= 1:
#                     files[fname] = arr1d.astype(np.float32)
#                 break
#     return files

# def preprocess(d_path, noise=False, snr=0, seed=42):
#     rng = np.random.RandomState(seed)
#     files = capture_de_signals(d_path)
#     if not files:
#         raise RuntimeError("No DE signals found in folder.")

#     train_folder = d_path + f"_TrainNoised_{snr}" if noise else d_path + "_Train"
#     test_folder = d_path + f"_TestNoised_{snr}" if noise else d_path + "_Test"
#     os.makedirs(train_folder, exist_ok=True)
#     os.makedirs(test_folder, exist_ok=True)

#     for fname, arr in files.items():
#         L = len(arr)
#         split_idx = (2 * L) // 3
#         train_seg = arr[:split_idx]
#         test_seg = arr[split_idx:]

#         if noise:
#             train_noised, train_noise = add_noise(train_seg, snr, rng=rng)
#             test_noised, test_noise = add_noise(test_seg, snr, rng=rng)
#         else:
#             train_noised, train_noise = train_seg.copy(), np.zeros_like(train_seg)
#             test_noised, test_noise = test_seg.copy(), np.zeros_like(test_seg)

#         # save with consistent varnames and base filename
#         base = os.path.splitext(fname)[0]
#         savemat(os.path.join(train_folder, base + ".mat"), {'DE': train_noised})
#         savemat(os.path.join(test_folder, base + ".mat"), {'DE': test_noised})

#     print(f"Saved train files to {train_folder} and test files to {test_folder}")

# if __name__ == "__main__":
#     path = r"D:\Bearing Project\CWRU\CWRU Case 08\QCNN for bearing diagnosis\data\0HP"
#     preprocess(d_path=path, noise=True, snr=0, seed=42)





# # data_split.py
# import os, numpy as np
# from scipy.io import loadmat, savemat

# def wgn(x, snr, rng=None):
#     if rng is None: rng = np.random.RandomState(None)
#     x = np.asarray(x).astype(np.float64)
#     x_flat = x.ravel()
#     p_signal = np.mean(x_flat ** 2)
#     p_noise = p_signal / (10 ** (snr / 10.0))
#     noise = rng.standard_normal(size=x_flat.shape) * np.sqrt(p_noise)
#     return noise.reshape(x.shape)

# def add_noise(data, snr, rng=None):
#     if rng is None: rng = np.random.RandomState(None)
#     return data + wgn(data, snr, rng=rng), wgn(data, snr, rng=rng)

# def capture_de_signals(d_path, min_length=2048):
#     files = {}
#     for fname in sorted(os.listdir(d_path)):
#         if not fname.endswith('.mat'): continue
#         try:
#             mat = loadmat(os.path.join(d_path, fname))
#         except Exception: continue
#         for key in mat:
#             if key.startswith("__"): continue
#             if 'de' in key.lower():
#                 arr = np.asarray(mat[key]).ravel()
#                 if arr.size >= min_length:
#                     files[fname] = arr.astype(np.float32)
#                 break
#     return files

# def preprocess(d_path, noise=False, snr=0, seed=42):
#     rng = np.random.RandomState(seed)
#     files = capture_de_signals(d_path)
#     if not files: raise RuntimeError("No DE signals found in folder.")

#     train_folder = f"{d_path}_TrainNoised_{snr}" if noise else f"{d_path}_Train"
#     test_folder = f"{d_path}_TestNoised_{snr}" if noise else f"{d_path}_Test"
#     os.makedirs(train_folder, exist_ok=True)
#     os.makedirs(test_folder, exist_ok=True)

#     for fname, arr in files.items():
#         L = len(arr)
#         split_idx = (2 * L) // 3
#         train_seg, test_seg = arr[:split_idx], arr[split_idx:]

#         if noise:
#             train_noised, _ = add_noise(train_seg, snr, rng=rng)
#             test_noised, _ = add_noise(test_seg, snr, rng=rng)
#         else:
#             train_noised, test_noised = train_seg.copy(), test_seg.copy()

#         base = os.path.splitext(fname)[0]
#         savemat(os.path.join(train_folder, base + ".mat"), {'DE': train_noised})
#         savemat(os.path.join(test_folder, base + ".mat"), {'DE': test_noised})

#     print(f"Saved train files to {train_folder} and test files to {test_folder}")

# if __name__ == "__main__":
#     path = r"D:\Bearing Project\CWRU\CWRU Case 08\QCNN for bearing diagnosis\data\0HP"
#     preprocess(d_path=path, noise=True, snr=0, seed=42)
