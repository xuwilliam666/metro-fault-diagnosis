import math
import torch
import torch.nn as nn
import torch.nn.functional as F


def my_kl_loss(p, q, eps=1e-8):
    """
    p, q: [B, H, T, T] probability distributions on last dim
    return: [B]  (batchwise KL)
    """
    p = p.clamp_min(eps)
    q = q.clamp_min(eps)
    res = p * (torch.log(p) - torch.log(q))
    # sum over last dim, then mean over query positions and heads
    return res.sum(dim=-1).mean(dim=-1).mean(dim=-1)  # [B]


class DataEmbedding(nn.Module):
    """Simple value embedding + positional embedding."""
    def __init__(self, c_in, d_model, dropout=0.0, max_len=5000):
        super().__init__()
        self.value_embedding = nn.Linear(c_in, d_model)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float32) * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0), persistent=False)  # [1, max_len, d_model]

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # x: [B, T, C]
        B, T, _ = x.shape
        x = self.value_embedding(x) + self.pe[:, :T, :]
        return self.dropout(x)


class AnomalyAttention(nn.Module):
    """
    Two-branch attention:
      - series: softmax(QK^T/sqrt(d))
      - prior : gaussian kernel based on |i-j| with learned sigma
    Returns:
      out: [B, T, d_model]
      series: [B, H, T, T]
      prior : [B, H, T, T]
      sigma : [B, H, T] (positive)
    """
    def __init__(self, d_model, n_heads, dropout=0.0):
        super().__init__()
        assert d_model % n_heads == 0
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads

        self.Wq = nn.Linear(d_model, d_model)
        self.Wk = nn.Linear(d_model, d_model)
        self.Wv = nn.Linear(d_model, d_model)

        # sigma per time point (and per head)
        self.Ws = nn.Linear(d_model, n_heads)  # -> [B,T,H]
        self.out_proj = nn.Linear(d_model, d_model)
        self.drop = nn.Dropout(dropout)

    def _gaussian_prior(self, T, sigma):
        """
        sigma: [B, H, T] positive
        build prior: [B, H, T, T]
        """
        device = sigma.device
        idx = torch.arange(T, device=device)
        dist = (idx[None, :] - idx[:, None]).abs().float()  # [T,T]

        # [1,1,T,T]
        dist = dist.view(1, 1, T, T)

        # sigma: [B,H,T] -> [B,H,T,1]
        s = sigma.unsqueeze(-1)  # [B,H,T,1]
        # gaussian kernel along key dimension
        prior = torch.exp(-(dist ** 2) / (2.0 * (s ** 2 + 1e-6)))
        prior = prior / (prior.sum(dim=-1, keepdim=True) + 1e-9)  # normalize
        return prior

    def forward(self, x):
        B, T, _ = x.shape

        q = self.Wq(x).view(B, T, self.n_heads, self.d_head).transpose(1, 2)  # [B,H,T,Dh]
        k = self.Wk(x).view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        v = self.Wv(x).view(B, T, self.n_heads, self.d_head).transpose(1, 2)

        attn_logits = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_head)  # [B,H,T,T]
        series = torch.softmax(attn_logits, dim=-1)
        series = self.drop(series)

        sigma = F.softplus(self.Ws(x)).transpose(1, 2) + 1e-6  # [B,H,T]
        prior = self._gaussian_prior(T, sigma)

        out = torch.matmul(series, v)  # [B,H,T,Dh]
        out = out.transpose(1, 2).contiguous().view(B, T, self.d_model)
        out = self.out_proj(out)

        return out, series, prior, sigma


class EncoderLayer(nn.Module):
    def __init__(self, d_model, n_heads, d_ff=256, dropout=0.0):
        super().__init__()
        self.attn = AnomalyAttention(d_model, n_heads, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
        )
        self.norm2 = nn.LayerNorm(d_model)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        a, series, prior, sigma = self.attn(x)
        x = self.norm1(x + self.drop(a))
        f = self.ff(x)
        x = self.norm2(x + self.drop(f))
        return x, series, prior, sigma


class AnomalyTransformer(nn.Module):
    """
    Returns:
      recon: [B,T,C]
      series_list: list of [B,H,T,T]
      prior_list : list of [B,H,T,T]
    """
    def __init__(self, c_in, d_model=128, n_heads=4, e_layers=3, d_ff=256, dropout=0.1):
        super().__init__()
        self.embedding = DataEmbedding(c_in, d_model, dropout=dropout)
        self.layers = nn.ModuleList([
            EncoderLayer(d_model, n_heads, d_ff=d_ff, dropout=dropout)
            for _ in range(e_layers)
        ])
        self.proj = nn.Linear(d_model, c_in)

    def forward(self, x):
        # x: [B,T,C]
        z = self.embedding(x)
        series_list, prior_list = [], []
        for layer in self.layers:
            z, series, prior, _ = layer(z)
            series_list.append(series)
            prior_list.append(prior)
        recon = self.proj(z)
        return recon, series_list, prior_list


def anomaly_transformer_loss(x, recon, series_list, prior_list, lambda_kl=1.0, eps=1e-8):
    """
    Paper minimax (two-phase) losses with stop-gradient.
    Returns:
      loss_min: update PRIOR branch (sigma/prior) -> minimize discrepancy (series detached)
      loss_max: update SERIES branch (attention/series) -> maximize discrepancy (prior detached)
      rec_loss, loss_series, loss_prior: for logging
    """
    # reconstruction (Frobenius/MSE)
    rec_loss = (recon - x).pow(2).mean()

    loss_series = 0.0  # KL(S || P_detach)
    loss_prior  = 0.0  # KL(P || S_detach)

    for s, p in zip(series_list, prior_list):
        s_ = s.clamp_min(eps)
        p_ = p.clamp_min(eps)

        # maximize discrepancy by updating S (P detached)
        loss_series = loss_series + (s_ * (torch.log(s_) - torch.log(p_.detach()))).sum(dim=-1).mean()

        # minimize discrepancy by updating P (S detached)
        loss_prior  = loss_prior  + (p_ * (torch.log(p_) - torch.log(s_.detach()))).sum(dim=-1).mean()

    # minimization phase: make PRIOR close to SERIES (series fixed)
    loss_min = rec_loss + lambda_kl * loss_prior

    # maximization phase: make SERIES far from PRIOR (prior fixed)
    loss_max = rec_loss - lambda_kl * loss_series

    return loss_min, loss_max, rec_loss, loss_series, loss_prior