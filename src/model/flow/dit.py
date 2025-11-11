import torch
import torch.nn.functional as F

from math import sqrt
from einops import rearrange
from torch import nn, Tensor


class AdaRMSNorm(nn.Module):
    def __init__(self, d_model: int, d_t: int, eps: float = 1e-6):
        super(AdaRMSNorm, self).__init__()

        self.beta = nn.Linear(d_t, d_model, bias=False)
        self.gamma = nn.Linear(d_t, d_model, bias=False)

        self.norm = nn.RMSNorm(d_model, eps=eps, elementwise_affine=False)

    def forward(self, x: Tensor, t: Tensor) -> Tensor:
        return self.gamma(t).unsqueeze(1) * self.norm(x) + self.beta(t).unsqueeze(1)


class SwiGLU(nn.Module):
    def __init__(self, d_model: int):
        super(SwiGLU, self).__init__()

        d_hidden = 4 * d_model

        self.hidden = nn.Linear(d_model, d_hidden, bias=False)
        self.gate = nn.Linear(d_model, d_hidden, bias=False)

        self.out = nn.Linear(d_hidden, d_model)

    def forward(self, x: Tensor) -> Tensor:
        return self.out(
            F.silu(self.gate(x)) * self.hidden(x)
        )


class SinusoidalEmbedding(nn.Module):
    def __init__(self, d_model: int, theta: float = 1e4):
        super(SinusoidalEmbedding, self).__init__()

        assert d_model % 2 == 0

        self.register_buffer(
            "theta",
            1.0 / (theta ** (2 * torch.arange(d_model // 2) / d_model)),
            persistent=False
        )

        self.ffn = SwiGLU(d_model)

    def forward(self, x: Tensor) -> Tensor:
        x = torch.outer(x, self.theta)

        return self.ffn(torch.stack((
            x.cos(),
            x.sin()
        ), dim=-1).flatten(-2))


class Attention(nn.Module):
    def __init__(self, d_model: int, n_heads: int):
        super(Attention, self).__init__()

        self.n_heads = n_heads
        self.scale = sqrt(n_heads / d_model)

        self.W_q = nn.Linear(d_model, d_model, bias=False)
        self.W_k = nn.Linear(d_model, d_model, bias=False)
        self.W_v = nn.Linear(d_model, d_model, bias=False)

        self.W_o = nn.Linear(d_model, d_model, bias=False)

    def forward(self, x: Tensor) -> Tensor:
        q = rearrange(self.W_q(x), "b l (n d) -> b n l d", n=self.n_heads)
        k = rearrange(self.W_k(x), "b l (n d) -> b n l d", n=self.n_heads)
        v = rearrange(self.W_v(x), "b l (n d) -> b n l d", n=self.n_heads)

        attn = self.scale * (q @ k.transpose(-2, -1))

        return self.W_o(rearrange(
            F.softmax(attn, dim=-1) @ v,
            "b n l d -> b l (n d)"
        ))


class DiTBlock(nn.Module):
    def __init__(self, d_model: int, n_heads: int, d_t: int):
        super(DiTBlock, self).__init__()

        self.attn_norm = AdaRMSNorm(d_model, d_t)
        self.attn = Attention(d_model, n_heads)
        self.attn_alpha = nn.Linear(d_t, d_model, bias=False)

        self.ffn_norm = AdaRMSNorm(d_model, d_t)
        self.ffn = SwiGLU(d_model)
        self.ffn_alpha = nn.Linear(d_t, d_model, bias=False)

        with torch.no_grad():
            self.attn_alpha.weight.fill_(0.0)
            self.ffn_alpha.weight.fill_(0.0)

    def forward(self, x: Tensor, t: Tensor):
        x = x + self.attn_alpha(t).unsqueeze(1) * self.attn(self.attn_norm(x, t))

        return x + self.ffn_alpha(t).unsqueeze(1) * self.ffn(self.ffn_norm(x, t))


class DiT(nn.Module):
    def __init__(self, d_in: int, num_classes: int, d_model: int, d_t: int, n_layers: int, n_heads: int):
        super(DiT, self).__init__()
        self.c_emb = nn.Embedding(num_classes, d_model)
        self.t_emb = SinusoidalEmbedding(d_t)

        self.stem = nn.Linear(d_in, d_model)

        self.layers = nn.ModuleList([
            DiTBlock(d_model, n_heads, d_t)
            for _ in range(n_layers)
        ])

        self.head = nn.Linear(d_model, d_in)

    def forward(self, x: Tensor, c: Tensor, t: Tensor) -> Tensor:
        condition = self.c_emb(c) + self.t_emb(t)

        x = self.stem(x)

        for layer in self.layers:
            x = layer(x, condition)

        return self.head(x)
