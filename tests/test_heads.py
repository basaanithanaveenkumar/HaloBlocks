"""Tests for head blocks."""

import pytest
import torch

from haloblocks import (
    ClassificationHead,
    LanguageModelHead,
    TokenClassificationHead,
)


def test_classification_head_cls_pooling():
    head = ClassificationHead(d_model=32, num_classes=10)
    x = torch.randn(2, 5, 32)
    out = head(x)
    assert out.shape == (2, 10)


def test_classification_head_mean_pooling():
    head = ClassificationHead(d_model=32, num_classes=10, pooling="mean")
    x = torch.randn(2, 5, 32)
    out = head(x)
    assert out.shape == (2, 10)


def test_classification_head_mean_pooling_with_mask():
    head = ClassificationHead(d_model=16, num_classes=5, pooling="mean")
    x = torch.randn(2, 6, 16)
    mask = torch.ones(2, 6, dtype=torch.float)
    mask[0, 4:] = 0.0
    out = head(x, attention_mask=mask)
    assert out.shape == (2, 5)


def test_classification_head_invalid_pooling():
    with pytest.raises(ValueError):
        ClassificationHead(d_model=16, num_classes=5, pooling="max")


def test_language_model_head_shape():
    head = LanguageModelHead(d_model=32, vocab_size=100)
    x = torch.randn(2, 5, 32)
    out = head(x)
    assert out.shape == (2, 5, 100)


def test_language_model_head_tie_weights():
    import torch.nn as nn

    head = LanguageModelHead(d_model=16, vocab_size=50)
    emb = nn.Embedding(50, 16)
    head.tie_weights(emb.weight)
    assert head.linear.weight is emb.weight


def test_token_classification_head_shape():
    head = TokenClassificationHead(d_model=32, num_classes=7)
    x = torch.randn(2, 10, 32)
    out = head(x)
    assert out.shape == (2, 10, 7)
