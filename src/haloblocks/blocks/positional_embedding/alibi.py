import torch
import math

from haloblocks.core.block import Block
from haloblocks.core.registry import BlockRegistry

@BlockRegistry.register("alibi_positional_bias")
class AlibiPositionalBias(Block):
    """
    Attention with Linear Biases (ALiBi).

    Computes a static bias tensor that is added to attention scores,
    penalizing distant token pairs based on their relative distance.

    Args:
        num_heads (int): Number of attention heads.
        max_len (int, optional): Maximum sequence length to precompute bias.
        slope_factor (float, optional): Scaling factor for slopes.
            If None, uses default geometric sequence: 2^(-8/n) for each head.
    """
    def __init__(self, num_heads, max_len=2048, slope_factor=None):
        super().__init__()
        self.num_heads = num_heads
        self.max_len = max_len
        self.slope_factor = slope_factor

        # Compute slopes for each head
        self._build_slopes()
        # Precompute bias for all possible lengths up to max_len
        self._build_bias_cache(max_len)

    def _build_slopes(self):
        """Compute slopes for each head."""
        if self.slope_factor is not None:
            slopes = torch.tensor([self.slope_factor] * self.num_heads)
        else:
            # Default: 2^(-8 * (i+1) / num_heads) for i in 0..num_heads-1
            n = self.num_heads
            def get_slopes(n):
                def get_slopes_power_of_2(n):
                    start = 2 ** (-(2 ** -(math.log2(n) - 3)))
                    return [start * (2 ** (-i)) for i in range(n)]
                if math.log2(n).is_integer():
                    return get_slopes_power_of_2(n)
                else:
                    closest_power_of_2 = 2 ** math.floor(math.log2(n))
                    slopes_a = get_slopes_power_of_2(closest_power_of_2)
                    slopes_b = get_slopes_power_of_2(2 * closest_power_of_2)
                    slopes = slopes_a + slopes_b[0::2][:n - closest_power_of_2]
                    return slopes
            slopes = get_slopes(n)
        self.register_buffer('slopes', torch.tensor(slopes))

    def _build_bias_cache(self, max_len):
        """Precompute bias matrix for all possible (tgt, src) lengths up to max_len."""
        # positions: (max_len, max_len) relative distance
        # We'll compute a triangular bias where bias[i, j] = -|i - j| * slope
        # But ALiBi uses a constant bias per head for each relative distance.
        # We can precompute for each head the matrix of shape (max_len, max_len)
        # with value = -abs(pos_i - pos_j) * slope[head].
        # However, storing max_len*max_len*heads may be large. We'll compute on the fly
        # using broadcasting. For simplicity, we'll compute when needed.
        pass

    def forward(self, tgt_len, src_len, device=None, **kwargs):
        """
        Return the ALiBi bias tensor.

        Args:
            tgt_len (int): Length of target sequence.
            src_len (int): Length of source sequence.
            device (torch.device, optional): Device for the tensor.

        Returns:
            torch.Tensor: Bias of shape (num_heads, tgt_len, src_len) to be added to scores.
        """
        if device is None:
            device = self.slopes.device
        # Create relative distance matrix: (tgt_len, src_len)
        # positions for target and source
        pos_tgt = torch.arange(tgt_len, device=device).view(-1, 1)  # (tgt_len, 1)
        pos_src = torch.arange(src_len, device=device).view(1, -1)  # (1, src_len)
        # ALiBi uses absolute distance: |i - j|
        abs_dist = torch.abs(pos_tgt - pos_src).float()  # (tgt_len, src_len)
        # Expand slopes to heads: (num_heads, 1, 1)
        slopes = self.slopes.view(-1, 1, 1)
        bias = -slopes * abs_dist  # (num_heads, tgt_len, src_len)
        return bias
