# CLAUDE.md — Vision Transformers

This file provides guidance for AI assistants working in this repository.

## Project Overview

A minimal, educational PyTorch implementation of Vision Transformers (ViT) as described in:
> [An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale](https://arxiv.org/abs/2010.11929) (Dosovitskiy et al., 2021)

The codebase is intentionally small and focused. It implements the full ViT inference pipeline without training utilities, data loaders, or pretrained weights.

---

## Repository Structure

```
vision-transformers/
├── README.md              # One-line project description with paper link
├── requirements.txt       # Pinned Python dependencies
├── src/
│   ├── __init__.py
│   └── models/
│       ├── __init__.py
│       └── vit.py         # Full ViT implementation (~162 lines)
└── test/
    ├── __init__.py
    └── test_vit.py        # Forward-pass smoke test
```

There is no `setup.py`, `pyproject.toml`, or build system. The project is a flat Python module consumed directly via `PYTHONPATH`.

---

## Dependencies

| Package  | Version | Role                        |
|----------|---------|-----------------------------|
| torch    | 2.2.2   | Neural network framework    |
| numpy    | 1.26.4  | Numerical ops (via PyTorch) |

Install with:
```bash
pip install -r requirements.txt
```

pytest is not listed in `requirements.txt` but is required to run tests:
```bash
pip install pytest
```

---

## Running Tests

Tests are in `test/test_vit.py` and use pytest. Run from the repository root:

```bash
pytest test/
```

The test imports `src.models.vit` using a package-relative path, so pytest must be run from the repo root (not from inside `test/`). No `conftest.py` or `pytest.ini` is present.

The single test (`test_forward`) performs a forward-pass shape assertion — it does not check numerical correctness or train the model.

---

## Architecture

All model code lives in `src/models/vit.py`. The classes map directly to the components described in the ViT paper.

### Components (bottom-up)

| Class / Function    | Description |
|---------------------|-------------|
| `patchify()`        | Splits a batch of images into flattened patch sequences |
| `MSA`               | Multihead Self-Attention |
| `MLP`               | Two-layer feed-forward network with GELU activation |
| `TransformerBlock`  | Pre-LN block: LayerNorm → MSA → residual, LayerNorm → MLP → residual |
| `TransformerEncoder`| Sequential stack of `TransformerBlock` instances |
| `MLPHead`           | Linear projection of the `[CLS]` token to class logits |
| `VisionTransfomer`  | Top-level model (see note on typo below) |

### Forward Pass — Tensor Shape Walkthrough

Given: `image_size=32`, `patch_size=16`, `embed_dim=64`, `batch_size=1`, `channels=3`, `num_classes=10`

```
Input image          (1, 3, 32, 32)
After patchify()     (1, 4, 768)      # 4 patches of size 16×16×3=768
After lin_embed      (1, 4, 64)       # project each patch to embed_dim
Prepend CLS token    (1, 5, 64)       # sequence length = num_patches + 1
Add pos_embed        (1, 5, 64)       # learned positional embedding (seq_len+1 positions)
After TransformerEncoder (1, 5, 64)
MLPHead (CLS @ index 0)  (1, 10)      # classification logits
```

### Key Design Choices

- **Pre-LN (pre-norm)**: LayerNorm is applied *before* MSA and MLP (not after), which stabilizes training.
- **Learned positional embeddings**: `nn.Embedding` rather than fixed sinusoidal encodings.
- **CLS token extraction**: `MLPHead` slices `x[:, 0, :]` to extract only the `[CLS]` token.
- **Dropout**: Default rate of `0.2` applied in MSA (after output projection) and MLP (after each linear layer).
- **Temperature scaling**: Attention scores divided by `sqrt(head_dim)` as per the original paper.
- **Square images and square patches assumed**: `image_size % patch_size == 0` is asserted.

---

## Known Quirks

- **Typo in class name**: The main model class is `VisionTransfomer` (missing an `r`). Do **not** rename it without updating all imports — the test imports it by this name.
- **`mlp_dim` not defaulted**: `VisionTransfomer.__init__` requires `mlp_dim` as a positional argument, but `TransformerBlock` and `MLP` do not expose a default. The test uses `mlp_dim=64`.
- **`channels` parameter**: Defaults to `3` (RGB) in `VisionTransfomer` but is not passed through to `patchify()`. The channel count is baked into `patch_dim = channels * patch_size**2`.
- **No `dropout` parameter on `VisionTransfomer`**: Dropout rate is hardcoded to the `0.2` default inside `TransformerEncoder` / `TransformerBlock`.
- **Positional embedding uses `torch.arange`**: Called without a device argument, which may fail if the model is moved to GPU. Pass `torch.arange(...).to(x.device)` when extending.
- **Dropout applied twice in `MLP`**: Once after GELU and once after the second linear layer (after `fc2`). This is a minor deviation from the standard ViT MLP which only applies dropout after the first activation and after the second linear.

---

## Conventions

- **No training code**: This is inference-only. Adding a training loop, optimizer, or loss function is a natural next step.
- **No data loaders**: Images are expected as raw `(B, C, H, W)` float tensors.
- **Module-level imports only**: No lazy imports. `torch`, `nn`, and `F` are imported at the top of `vit.py`.
- **Class-based structure**: Every architectural component is an `nn.Module`. Prefer adding new components as modules rather than standalone functions.
- **Test placement**: Tests live in `test/` (not `tests/`). Keep new test files in that directory.

---

## Extending the Codebase

When adding features, follow these patterns:

1. **New model variants**: Subclass or compose existing modules in `src/models/`. Add new files (e.g., `deit.py`, `swin.py`) rather than modifying `vit.py`.
2. **New tests**: Add test files to `test/` and name them `test_<component>.py`. Use pytest assertions.
3. **Configurable dropout**: Thread a `dropout` kwarg through `VisionTransfomer` → `TransformerEncoder` → `TransformerBlock`.
4. **GPU support**: Fix positional embedding device placement in `VisionTransfomer.forward()` by changing `torch.arange(self.seq_length+1)` to `torch.arange(self.seq_length+1, device=x.device)`.
5. **Dependencies**: Add new packages to `requirements.txt` with pinned versions.

---

## Git Workflow

- Default development branch: `master`
- Feature/task branches follow the pattern: `claude/<description>-<id>`
- Remote: `origin` (single remote)
- No CI/CD configuration is present in the repository.
