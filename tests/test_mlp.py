import torch
import torch.nn as nn

import haloblocks as hb
from haloblocks import blocks


def test_mlp_basic():
    print("Testing Basic MLP...")
    # 2-layer MLP: 128 -> 256 -> 64
    mlp = hb.create("MLP", input_dim=128, hidden_dims=[256, 64])
    x = torch.randn(2, 10, 128)
    out = mlp(x)
    assert out.shape == (2, 10, 64)
    print("Success")


def test_mlp_advanced():
    print("Testing Advanced MLP (No Bias, GELU, Final Activation)...")
    # 3-layer MLP: 64 -> 128 -> 128 -> 32
    mlp = blocks.MLP(
        input_dim=64, hidden_dims=[128, 128], output_dim=32, activation="gelu", bias=False, last_layer_activation=True
    )

    # Check bias
    for m in mlp.modules():
        if isinstance(m, nn.Linear):
            assert m.bias is None

    # Check final activation (last layer should be GELU)
    assert isinstance(mlp.model[-1], nn.GELU)

    x = torch.randn(1, 64)
    out = mlp(x)
    assert out.shape == (1, 32)
    print("Success")


def test_mlp_errors():
    print("Testing MLP Error Handling...")
    try:
        hb.create("MLP", input_dim=10, hidden_dims=[20], activation="invalid_act")
        assert False, "Should have raised ValueError"
    except ValueError as e:
        print(f"Caught expected error: {e}")


if __name__ == "__main__":
    try:
        test_mlp_basic()
        test_mlp_advanced()
        test_mlp_errors()
        print("\nAll MLP tests passed!")
    except Exception as e:
        print(f"\nTests failed: {e}")
        import traceback

        traceback.print_exc()
        exit(1)
