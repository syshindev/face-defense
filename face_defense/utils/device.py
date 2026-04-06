import torch


def get_device(device_str: str = "auto") -> torch.device:
    # Resolve device string to torch.device
    if device_str == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda:0")
        return torch.device("cpu")
    return torch.device(device_str)


def get_device_info() -> dict:
    # Return GPU/CPU info for logging
    info = {"device": "cpu", "gpu_name": None, "gpu_count": 0}
    if torch.cuda.is_available():
        info["device"] = "cuda"
        info["gpu_name"] = torch.cuda.get_device_name(0)
        info["gpu_count"] = torch.cuda.device_count()
    return info
