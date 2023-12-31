import torch

if __name__ == "__main__":
    if torch.cuda.is_available():
        device_name = torch.cuda.get_device_name(1)
        print("GPU Device Name:", device_name)
    else:
        print("No GPU available")