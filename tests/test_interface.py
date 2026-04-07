import torch

import haloblocks as hb
from haloblocks import blocks


def test_interface():
    print("Testing Keyword-style create...")
    attn1 = hb.create("MultiHeadAttention", emb_dim=128, num_heads=4)
    assert attn1.emb_dim == 128
    print("Success")

    print("Testing Dictionary-style create...")
    attn2 = hb.create({"type": "MultiHeadAttention", "emb_dim": 128, "num_heads": 4})
    assert attn2.emb_dim == 128
    print("Success")

    print("Testing Layers proxy style...")
    attn3 = blocks.MultiHeadAttention(emb_dim=128, num_heads=4)
    assert attn3.emb_dim == 128
    print("Success")

    print("Testing forward pass...")
    x = torch.randn(1, 10, 128)
    out = attn3(x)
    assert out.shape == (1, 10, 128)
    print("Success")


if __name__ == "__main__":
    try:
        test_interface()
        print("\nAll interface tests passed!")
    except Exception as e:
        print(f"\nTests failed: {e}")
        exit(1)
