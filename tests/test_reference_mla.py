"""
Test script for DeepSeek-V2-Lite MLA reference implementation.

Tests:
  1. Dimension test: verifies output shape is [bsz, seq, 2048]
  2. Stability test: verifies no NaNs or Infs in output
  3. Determinism test: same input always gives same output
  4. Causal mask test: token at position i does NOT attend to position i+1
"""

import sys
import os
import torch

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))
from reference.mla import DeepSeekMLA


def get_device_and_dtype():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    print(f"Using device: {device}, dtype: {dtype}")
    return device, dtype


def test_mla_dimensions():
    print("\nRunning MLA Dimension Test...")
    device, dtype = get_device_and_dtype()
    bsz, seq_len, hidden_size = 2, 16, 2048

    mla = DeepSeekMLA().to(device).to(dtype)
    x = torch.randn(bsz, seq_len, hidden_size, device=device, dtype=dtype)

    with torch.no_grad():
        out = mla(x)

    assert out.shape == (bsz, seq_len, hidden_size), \
        f"Shape mismatch: expected {(bsz, seq_len, hidden_size)}, got {out.shape}"
    print(f"Output shape: {out.shape} ")
    print("MLA Dimension Test Passed!")


def test_mla_stability():
    print("\nRunning MLA Stability Test (NaN/Inf check)...")
    device, dtype = get_device_and_dtype()
    mla = DeepSeekMLA().to(device).to(dtype)
    x = torch.randn(1, 32, 2048, device=device, dtype=dtype)

    with torch.no_grad():
        out = mla(x)

    assert not torch.isnan(out).any(), "Found NaNs in output"
    assert not torch.isinf(out).any(), "Found Infs in output"
    print("MLA Stability Test Passed!")


def test_mla_determinism():
    print("\nRunning MLA Determinism Test...")
    device, dtype = get_device_and_dtype()
    mla = DeepSeekMLA().to(device).to(dtype)
    x = torch.randn(1, 8, 2048, device=device, dtype=dtype)

    with torch.no_grad():
        out1 = mla(x)
        out2 = mla(x)

    assert torch.allclose(out1, out2), "Non-deterministic output detected!"
    print("MLA Determinism Test Passed!")


def test_mla_causality():
    """
    Verifies causality: changing token at position i should NOT affect 
    the output at positions 0..i-1.
    """
    print("\nRunning MLA Causality Test...")
    device, dtype = get_device_and_dtype()

    # Use float32 for this test regardless — easier to compare precisely
    mla = DeepSeekMLA().to(device).to(torch.float32)

    torch.manual_seed(0)
    x = torch.randn(1, 8, 2048, device=device, dtype=torch.float32)
    x_modified = x.clone()
    # Modify the last token
    x_modified[:, -1, :] = torch.randn(1, 2048, device=device, dtype=torch.float32)

    with torch.no_grad():
        out_orig = mla(x)
        out_mod = mla(x_modified)

    # The outputs at positions 0..6 should be identical
    # because they cannot attend to position 7 (causal mask)
    assert torch.allclose(out_orig[:, :-1, :], out_mod[:, :-1, :], atol=1e-5), \
        "Causality violated: earlier token outputs changed when a later token was modified!"
    print("MLA Causality Test Passed!")


if __name__ == "__main__":
    test_mla_dimensions()
    test_mla_stability()
    test_mla_determinism()
    test_mla_causality()
    print("\n All MLA Reference Tests Passed!")
