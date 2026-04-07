import torch
import torch.nn as nn
import torch.nn.functional as F

from haloblocks.core.block import Block
from haloblocks.core.registry import BlockRegistry


class Expert(Block):
    """
    A single expert layer within a Mixture-of-Experts (MoE) block.

    This implementation uses a SwiGLU activation function:
    (Swish(xW1) * xW3) * W2.

    Args:
        emb_dim (int): Input and output dimensionality.
        hid_dim (int): Intermediate dimensionality for the SwiGLU layer.
        dropout (float): Dropout probability. Defaults to 0.0.
    """

    def __init__(self, emb_dim, hid_dim, dropout=0.0):
        super().__init__()
        self.w1 = nn.Linear(emb_dim, hid_dim, bias=False)
        self.w2 = nn.Linear(hid_dim, emb_dim, bias=False)
        self.w3 = nn.Linear(emb_dim, hid_dim, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, **kwargs):
        # SwiGLU activation: (Swish(xW1) * xW3) * W2
        return self.dropout(self.w2(F.silu(self.w1(x)) * self.w3(x)))


class NoiseBestKRouter(Block):
    """
    A router that selects the best K experts with added Gaussian noise for load balancing.

    Args:
        emb_dim (int): Dimensionality of the input embeddings.
        num_exprts (int): Total number of experts to route between.
        best_k (int): Number of top experts to select for each token.
    """

    def __init__(self, emb_dim, num_exprts, best_k):
        super().__init__()
        self.best_k = best_k
        self.bestk_layer = nn.Linear(emb_dim, num_exprts)
        self.noise_linear = nn.Linear(emb_dim, num_exprts)

    def forward(self, x, **kwargs):
        """
        Routes inputs to the best K experts.

        Args:
            x (torch.Tensor): Input tensor of shape (batch, seq_len, emb_dim).

        Returns:
            Tuple[torch.Tensor, torch.Tensor]:
                - router_output: Softmax probabilities over chosen experts (sparse).
                - idxs: Indices of the top-k experts.
        """
        logits = self.bestk_layer(x)
        noise_logits = self.noise_linear(x)
        # add the gaussian noise to logits
        if self.training:
            noise = torch.randn_like(logits) * F.softplus(noise_logits)
            noisy_logits = logits + noise
        else:
            noisy_logits = logits

        best_k_logits, idxs = noisy_logits.topk(self.best_k, dim=-1)
        zeros = torch.full_like(noisy_logits, float("-inf"))
        sparse_logits = zeros.scatter(-1, idxs, best_k_logits)
        router_output = F.softmax(sparse_logits, dim=-1)
        return router_output, idxs


@BlockRegistry.register()
class DeepseekMoE(Block):
    """
    DeepSeek Mixture-of-Experts (MoE) implementation.

    This architecture combines 'shared experts' (always active) with 'routed experts'
    (selected by a router).

    Args:
        emb_dim (int): Input and output dimensionality.
        hid_dim (int): Intermediate dimensionality for each expert.
        num_router_exprts (int): Total number of experts available for routing.
        best_k (int): Number of routed experts to activate per token.
        num_shared_exprts (int): Number of shared experts that are always active.
    """

    def __init__(self, emb_dim, hid_dim, num_router_exprts, best_k, num_shared_exprts):
        super().__init__()
        self.router = NoiseBestKRouter(emb_dim, num_router_exprts, best_k)
        self.shared_experts = nn.ModuleList([Expert(emb_dim, hid_dim) for _ in range(num_shared_exprts)])
        self.routed_experts = nn.ModuleList([Expert(emb_dim, hid_dim) for _ in range(num_router_exprts)])
        self.best_k = best_k

    def forward(self, x, **kwargs):
        """
        Computes the MoE forward pass.

        Args:
            x (torch.Tensor): Input tensor of shape (batch, seq_len, emb_dim).
            **kwargs: Ignored.

        Returns:
            torch.Tensor: Aggregated output from shared and routed experts.
        """
        batch, seq, dim = x.shape
        x_flat = x.view(-1, dim)
        shared_output = 0
        for expert in self.shared_experts:
            shared_output += expert(x_flat)

        gating_output, idxs = self.router(x)
        final_output = torch.zeros_like(x)
        flat_x = x.view(-1, x.size(-1))
        flat_gating_output = gating_output.view(-1, gating_output.size(-1))

        for i, expert in enumerate(self.routed_experts):
            expert_mask = (idxs == i).any(dim=-1)
            flat_mask = expert_mask.view(-1)

            if flat_mask.any():
                expert_input = flat_x[flat_mask]
                expert_output = expert(expert_input)
                gating_scores = flat_gating_output[flat_mask, i].unsqueeze(1)
                weighted_output = expert_output * gating_scores
                final_output[expert_mask] += weighted_output.squeeze(1)
        return final_output + shared_output.view(batch, seq, dim)
