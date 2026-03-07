import torch
from torch import nn
from torch.nn import functional as F


def patchify(images, patch_size):
  """
  Chunks a batch of images into patches according to the ViT paper.

  Args:
    images: A PyTorch tensor of shape (batch_size, channels, height, width).
    patch_size: An integer representing the size of each patch (P).

  Returns:
    A PyTorch tensor of shape (batch_size, num_patches, patch_dim), where
    patch_dim = patch_size * patch_size * channels.
  """

  batch_size, _, height, width = images.shape
  patches = images.unfold(2, patch_size, patch_size).unfold(3, patch_size, patch_size)
  num_patches_h = height // patch_size
  num_patches_w = width // patch_size
  patches = patches.permute(0, 2, 3, 1, 4, 5).reshape(batch_size, num_patches_h * num_patches_w, -1)

  return patches   

class MSA(nn.Module):
    """
    Multihead self-attention layer
    """
    def __init__(self, embed_dim, n_heads, dropout=0.2):
        super().__init__()
        self.embed_dim = embed_dim
        self.n_heads = n_heads

        assert self.embed_dim % self.n_heads == 0

        self.head_dim = self.embed_dim // self.n_heads

        self.q_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.k_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.v_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, S = x.shape[0], x.shape[1]
        q, k, v = self.q_proj(x), self.k_proj(x), self.v_proj(x) # each B x S x D
        q, k, v = [
            m.reshape(B, S, self.n_heads, self.head_dim).transpose(1,2) 
            for m in [q, k, v]
        ]

        scores = q @ k.transpose(-1,-2) / (self.head_dim)**0.5
        scores = F.softmax(scores, dim=-1)
        out = scores @ v
        out = out.transpose(1,2).reshape(B, S, -1)

        out = self.dropout(self.proj(out))
        return out

class MLP(nn.Module):
    """
    Multi-layer Perceptron
    """
    def __init__(self, embed_dim, mlp_dim, dropout=0.2):
        super().__init__()
        self.fc1 = nn.Linear(embed_dim, mlp_dim)
        self.fc2 = nn.Linear(mlp_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.fc1(x)
        x = F.gelu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x

class TransformerBlock(nn.Module):
    def __init__(self, embed_dim, n_heads, mlp_dim, dropout=0.2):
        super().__init__()
        self.msa = MSA(embed_dim, n_heads, dropout)
        self.mlp = MLP(embed_dim, mlp_dim, dropout)
        self.ln1 = nn.LayerNorm(embed_dim)
        self.ln2 = nn.LayerNorm(embed_dim)

    def forward(self, x):
        x = x + self.msa(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x

class TransformerEncoder(nn.Module):
    def __init__(self, embed_dim, n_heads, mlp_dim, num_blocks, dropout=0.2):
        super().__init__()
        self.blocks = nn.ModuleList([
            TransformerBlock(embed_dim, n_heads, mlp_dim, dropout)
            for _ in range(num_blocks)
        ])

    def forward(self, x):
        for block in self.blocks:
            x = block(x)
        return x

class MLPHead(nn.Module):
    def __init__(self, embed_dim, num_classes):
        super().__init__()
        self.fc = nn.Linear(embed_dim, num_classes)

    def forward(self, x):
        x = self.fc(x[:,0,:])
        return x

class VisionTransfomer(nn.Module):

  def __init__(
      self,
      image_size,
      patch_size,
      embed_dim,
      num_heads,
      num_blocks,
      num_classes,
      mlp_dim,
      channels = 3,
      dropout = 0.2
  ):
    super().__init__()
    # I assume square images, square patches for simplicity.

    assert image_size % patch_size == 0
    self.seq_length = (image_size // patch_size)**2
    self.embed_dim = embed_dim
    self.patch_size = patch_size
    self.patch_dim = channels * patch_size**2 # assume 3 channels per image?

    # Should these layers have bias?
    self.lin_embed = nn.Linear(self.patch_dim, self.embed_dim)
    self.pos_embed = nn.Embedding(self.seq_length+1, self.embed_dim) # one more for cls token?
    self.cls_token = nn.Parameter(torch.randn(1, 1, self.embed_dim))

    self.encoder = TransformerEncoder(
        embed_dim=self.embed_dim,
        n_heads=num_heads,
        mlp_dim=mlp_dim,
        num_blocks=num_blocks,
        dropout=dropout
    )
    self.head = MLPHead(self.embed_dim, num_classes)

  def forward(self, x):
    x = patchify(x, self.patch_size) # B x seq_length x patch_dim

    l_embed = self.lin_embed(x)
    seq_embed = torch.cat(
        [torch.tile(self.cls_token, dims=(x.shape[0],1,1)), l_embed],
        axis=1
    )

    pos_embed = self.pos_embed(torch.arange(self.seq_length+1, device=x.device))
    x = seq_embed + pos_embed # B x seq_length x embed_dim
    x = self.encoder(x)

    return self.head(x)