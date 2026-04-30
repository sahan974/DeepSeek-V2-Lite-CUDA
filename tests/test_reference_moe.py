import torch
import sys
import os

# Add src to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

from reference.moe import DeepSeekMoE

def test_moe_dimensions():
    print("Running MoE Dimension Test...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    
    print(f"Using device: {device}, dtype: {dtype}")
    
    batch_size = 2
    seq_len = 32
    hidden_size = 2048
    
    moe = DeepSeekMoE(
        hidden_size=hidden_size,
        moe_intermediate_size=1408,
        n_routed_experts=64,
        num_experts_per_tok=6,
        n_shared_experts=2
    ).to(device).to(dtype)
    
    x = torch.randn(batch_size, seq_len, hidden_size).to(device).to(dtype)
    
    output = moe(x)
    
    assert output.shape == x.shape, f"Shape mismatch: {output.shape} vs {x.shape}"
    print("MoE Dimension Test Passed!")

def test_moe_weight_stability():
    print("Running MoE Weight Stability Test...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    
    moe = DeepSeekMoE().to(device).to(dtype)
    x = torch.randn(1, 1, 2048).to(device).to(dtype)
    
    output = moe(x)
    assert not torch.isnan(output).any(), "Found NaNs in output"
    assert not torch.isinf(output).any(), "Found Infs in output"
    print("MoE Stability Test Passed!")

if __name__ == "__main__":
    test_moe_dimensions()
    test_moe_weight_stability()
