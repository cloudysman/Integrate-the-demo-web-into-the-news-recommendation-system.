import torch

if torch.cuda.is_available():
    print("CUDA is available! Your system supports GPU acceleration.")
    print(f"Number of GPU devices: {torch.cuda.device_count()}")
    print(f"Current GPU device: {torch.cuda.get_device_name(0)}")
else:
    print("CUDA is not available. Your system does not support GPU acceleration.")
