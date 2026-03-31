"""
torch-vulkan: Vulkan compute backend for PyTorch

Usage:
    import torch
    import torch_vulkan

    # Check availability
    print(torch_vulkan.is_available())
    print(torch_vulkan.device_name())

    # Create tensors on Vulkan device
    a = torch.randn(64, 64, device="vulkan")
    b = torch.randn(64, 64, device="vulkan")
    c = torch.mm(a, b)  # runs matmul.spv on GPU

    # Or move existing tensors
    x = torch.randn(128, 128)
    x_vk = x.vulkan()
"""

import os
import torch

# Load the C++ extension — this registers PrivateUse1 as "vulkan"
from torch_vulkan import _C

# Point the shader loader at our bundled .spv files
_shader_dir = os.path.join(os.path.dirname(__file__), "..", "csrc", "shaders")
if os.path.isdir(_shader_dir):
    _C._set_shader_dir(os.path.abspath(_shader_dir))


def is_available() -> bool:
    """Check if a Vulkan-capable GPU is available."""
    return _C._is_available()


def device_name() -> str:
    """Return the name of the Vulkan device."""
    return _C._device_name()


def device(index: int = 0) -> torch.device:
    """Return a torch.device for the Vulkan backend."""
    return torch.device("vulkan", index)
