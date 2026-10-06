import sys
import platform
import torch

print("="*40)
print("HARDWARE & SOFTWARE ENVIRONMENT")
print("="*40)

# 1. OS & Python
print(f"Operating System : {platform.system()} {platform.release()}")
print(f"Python Version   : {sys.version.split()[0]}")

# 2. PyTorch & CUDA
print(f"PyTorch Version  : {torch.__version__}")
if torch.cuda.is_available():
    print(f"CUDA Toolkit     : {torch.version.cuda}")
    print(f"cuDNN Version    : {torch.backends.cudnn.version()}")
    
    # 3. GPU Info
    gpu_name = torch.cuda.get_device_name(0)
    vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
    print(f"GPU Model        : {gpu_name}")
    print(f"GPU VRAM         : {vram_gb:.1f} GB")
else:
    print("GPU NOT DETECTED (Running on CPU)")
print("="*40)