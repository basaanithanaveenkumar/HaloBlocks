import torch
import torch.nn as nn
from typing import List, Union, Optional
from haloblocks.core.block import Block
from haloblocks.core.registry import BlockRegistry

@BlockRegistry.register("mlp")
class MLP(Block):
    """
    A configurable Multi-Layer Perceptron (MLP) block.

    Args:
        input_dim (int): Dimensionality of the input.
        hidden_dims (List[int]): List of hidden layer dimensions.
        output_dim (Optional[int]): Dimensionality of the output. 
            If None, the last value in `hidden_dims` is used as output dimension.
        activation (str): Activation function to use between layers. 
            Supported: 'relu', 'gelu', 'tanh', 'silu'. Defaults to 'relu'.
        bias (bool): Whether to include bias in linear layers. Defaults to True.
        last_layer_activation (bool): If True, applies activation to the final layer.
            Defaults to False (final layer is strictly linear).
        dropout (float): Dropout probability applied after each activation.
    """
    
    _activations = {
        'relu': nn.ReLU,
        'gelu': nn.GELU,
        'tanh': nn.Tanh,
        'silu': nn.SiLU
    }

    def __init__(
        self, 
        input_dim: int, 
        hidden_dims: List[int], 
        output_dim: Optional[int] = None,
        activation: str = 'relu',
        bias: bool = True,
        last_layer_activation: bool = False,
        dropout: float = 0.0
    ):
        super().__init__()
        
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.output_dim = output_dim or hidden_dims[-1]
        self.activation_name = activation.lower()
        self.use_bias = bias
        self.last_layer_activation = last_layer_activation
        
        if self.activation_name not in self._activations:
            raise ValueError(f"Unsupported activation: {activation}. Supported: {list(self._activations.keys())}")
        
        activation_cls = self._activations[self.activation_name]
        
        layers = []
        curr_dim = input_dim
        
        # Build hidden layers
        for i, h_dim in enumerate(hidden_dims):
            # If this is the last hidden layer and no explicit output_dim, check if we apply activation
            is_last = (i == len(hidden_dims) - 1) and (output_dim is None)
            
            layers.append(nn.Linear(curr_dim, h_dim, bias=bias))
            
            if not is_last or last_layer_activation:
                layers.append(activation_cls())
                if dropout > 0:
                    layers.append(nn.Dropout(dropout))
            
            curr_dim = h_dim
        
        # Add output layer if explictly defined
        if output_dim is not None:
            layers.append(nn.Linear(curr_dim, output_dim, bias=bias))
            if last_layer_activation:
                layers.append(activation_cls())
                if dropout > 0:
                    layers.append(nn.Dropout(dropout))

        self.model = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        """
        Forward pass of the MLP.

        Args:
            x (torch.Tensor): Input tensor of shape [..., input_dim].
            **kwargs: Ignored.

        Returns:
            torch.Tensor: Output tensor of shape [..., output_dim].
        """
        return self.model(x)

    def __repr__(self):
        return (f"MLP(input_dim={self.input_dim}, hidden_dims={self.hidden_dims}, "
                f"output_dim={self.output_dim}, activation={self.activation_name}, "
                f"bias={self.use_bias}, last_activation={self.last_layer_activation})")
