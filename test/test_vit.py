import torch
from src.models.vit import VisionTransfomer

def test_forward():
    model = VisionTransfomer(
            image_size=32,
            patch_size=16,
            embed_dim=64,
            num_heads=4,
            num_blocks=2,
            num_classes=10,
            mlp_dim=64
    )
    batch = torch.randn(1, 3, 32, 32)
    y = model(batch)
    assert y.shape == (1, 10)