import torch


def get_device(device=None):
    """Resolve a device string to a torch.device.

    Args:
        device: One of "auto", "cpu", "cuda", or None.
            - None or "auto": use CUDA if available, else CPU.
            - "cpu": force CPU.
            - "cuda": force CUDA (raises error if unavailable).

    Returns:
        A string suitable for torch tensor/model .to() calls.
    """
    if device is None or device == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA device requested but torch.cuda.is_available() is False. "
            "Install a CUDA-enabled PyTorch build or use device='cpu'."
        )
    return device
