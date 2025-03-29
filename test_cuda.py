import torch

def check_gpu():
    if torch.cuda.is_available():
        print("GPU is available")
        print(f"Using {torch.cuda.get_device_name(0)}")
    else:
        print("GPU is not available")

check_gpu()