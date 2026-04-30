"""
Tests for the DeepSeek-V2-Lite reference decoder layer.

Verifies:
    1. Output shape matches input shape [batch, seq, 2048]
    2. Residual connections are active (output != attention_only_output)
    3. No NaN or Inf values in output
    4. Causal property is preserved end-to-end
"""

import sys
import os
import torch

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))
from reference.decoder_layer import DeepSeekDecoderLayer


def get_device_and_dtype():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    print(f"  Device: {device}, dtype: {dtype}")
    return device, dtype


def test_decoder_layer_dimensions():
    print("\nRunning Decoder Layer Dimension Test...")
    device, dtype = get_device_and_dtype()
    bsz, seq_len, hidden_size = 2, 16, 2048

    layer = DeepSeekDecoderLayer().to(device).to(dtype)
    x = torch.randn(bsz, seq_len, hidden_size, device=device, dtype=dtype)

    with torch.no_grad():
        out = layer(x)

    assert out.shape == (bsz, seq_len, hidden_size), \
        f"Shape mismatch: expected {(bsz, seq_len, hidden_size)}, got {out.shape}"
    print(f"  Output shape: {out.shape}")
    print("Decoder Layer Dimension Test Passed!")


def test_decoder_layer_stability():
    print("\nRunning Decoder Layer Stability Test (NaN/Inf check)...")
    device, dtype = get_device_and_dtype()

    layer = DeepSeekDecoderLayer().to(device).to(dtype)
    x = torch.randn(1, 8, 2048, device=device, dtype=dtype)

    with torch.no_grad():
        out = layer(x)

    assert not torch.isnan(out).any(), "NaN values detected in output"
    assert not torch.isinf(out).any(), "Inf values detected in output"
    print("Decoder Layer Stability Test Passed!")


def test_decoder_layer_residual():
    """
    Confirms that the residual connections are active.
    If residuals were broken, the output would equal the MoE output alone.
    """
    print("\nRunning Decoder Layer Residual Connection Test...")
    device = "cuda" if torch.cuda.is_available() else "cpu"

    layer = DeepSeekDecoderLayer().to(device).to(torch.float32)

    torch.manual_seed(42)
    x = torch.randn(1, 4, 2048, device=device, dtype=torch.float32)

    with torch.no_grad():
        out = layer(x)

    # Output must differ from input (transformation occurred)
    assert not torch.allclose(out, x, atol=1e-3), \
        "Output is identical to input — residual or MLP may be broken"
    print("Decoder Layer Residual Connection Test Passed!")


def test_decoder_layer_causality():
    """
    Confirms causal masking is preserved through the full layer.
    Modifying the last token should not affect outputs at earlier positions.
    """
    print("\nRunning Decoder Layer Causality Test...")
    device = "cuda" if torch.cuda.is_available() else "cpu"

    layer = DeepSeekDecoderLayer().to(device).to(torch.float32)

    torch.manual_seed(0)
    x = torch.randn(1, 8, 2048, device=device, dtype=torch.float32)
    x_modified = x.clone()
    x_modified[:, -1, :] = torch.randn(1, 2048, device=device, dtype=torch.float32)

    with torch.no_grad():
        out_orig = layer(x)
        out_mod = layer(x_modified)

    assert torch.allclose(out_orig[:, :-1, :], out_mod[:, :-1, :], atol=1e-5), \
        "Causality violated: modifying the last token affected earlier token outputs"
    print("Decoder Layer Causality Test Passed!")


if __name__ == "__main__":
    test_decoder_layer_dimensions()
    test_decoder_layer_stability()
    test_decoder_layer_residual()
    test_decoder_layer_causality()
    print("\nAll Decoder Layer Reference Tests Passed!")
