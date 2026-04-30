import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple

class ExpertMLP(nn.Module):
    """
    Standard SwiGLU MLP used in DeepSeek-V2 experts.
    Geometry: hidden_size -> moe_intermediate_size -> hidden_size
    Logic: down_proj(SiLU(gate_proj(x)) * up_proj(x))
    """
    def __init__(self, hidden_size: int, intermediate_size: int):
        super().__init__()
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))

class MoEGate(nn.Module):
    """
    DeepSeek-V2-Lite Gate using Softmax and Greedy Top-K.
    """
    def __init__(self, hidden_size: int, n_routed_experts: int, top_k: int):
        super().__init__()
        self.top_k = top_k
        self.n_routed_experts = n_routed_experts
        self.weight = nn.Parameter(torch.empty(n_routed_experts, hidden_size))
        # Initialized properly in official model, random initialization is used for reference
        nn.init.kaiming_uniform_(self.weight, a=5**0.5)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # x shape: [batch * seq, hidden_size]
        logits = F.linear(x.float(), self.weight.float(), None)
        scores = logits.softmax(dim=-1)
        
        # Greedy Top-K
        topk_weight, topk_idx = torch.topk(scores, k=self.top_k, dim=-1, sorted=False)
        
        # DeepSeek-V2-Lite does NOT normalize top-k weights by default
        # but applies a routed_scaling_factor (usually 1.0)
        return topk_idx, topk_weight

class DeepSeekMoE(nn.Module):
    """
    Top-level Mixture of Experts module for DeepSeek-V2-Lite.
    Combines Routed Experts and Shared Experts.
    """
    def __init__(self, 
                 hidden_size: int = 2048, 
                 moe_intermediate_size: int = 1408, 
                 n_routed_experts: int = 64, 
                 num_experts_per_tok: int = 6,
                 n_shared_experts: int = 2):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_experts_per_tok = num_experts_per_tok
        
        # Gating mechanism
        self.gate = MoEGate(hidden_size, n_routed_experts, num_experts_per_tok)
        
        # Routed Experts
        self.experts = nn.ModuleList([
            ExpertMLP(hidden_size, moe_intermediate_size) 
            for _ in range(n_routed_experts)
        ])
        
        # Shared Experts (always active)
        # In DeepSeek-V2, shared experts are often implemented as one large MLP
        # with intermediate_size = moe_intermediate_size * n_shared_experts
        if n_shared_experts > 0:
            self.shared_experts = ExpertMLP(hidden_size, moe_intermediate_size * n_shared_experts)
        else:
            self.shared_experts = None

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        orig_shape = hidden_states.shape
        # Flatten batch and seq dimensions
        x = hidden_states.view(-1, self.hidden_size)
        
        # 1. Compute Gating
        topk_idx, topk_weight = self.gate(x)
        
        # 2. Routed Experts Computation (Reference implementation uses a loop)
        # This part is targeted for optimization with custom CUDA kernels
        final_output = torch.zeros_like(x)
        
        # For each expert, find which tokens are assigned to it
        for i, expert in enumerate(self.experts):
            # Find tokens where this expert is in the top-k
            mask = (topk_idx == i)
            if mask.any():
                # Get the indices of tokens assigned to this expert
                token_indices = mask.any(dim=1).nonzero().flatten()
                # Find which of the top-k positions this expert occupies for each token
                # (Used to get the corresponding weight)
                weight_mask = (topk_idx[token_indices] == i)
                expert_weights = topk_weight[token_indices][weight_mask].unsqueeze(-1)
                
                # Compute expert output and scale by weight
                expert_out = expert(x[token_indices])
                final_output[token_indices] += expert_out * expert_weights
        
        # 3. Shared Experts Computation (Active for ALL tokens)
        if self.shared_experts is not None:
            final_output += self.shared_experts(x)
            
        return final_output.view(*orig_shape)
