import torch


def _active_lora_index(self) -> int:
    active = torch.nonzero(self.adapter_enabled, as_tuple=False).flatten()
    if active.numel() == 0:
        return -1
    return int(active[0].item())


def _expert_linear(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    weight = weight.contiguous()
    if weight.shape[0] == x.shape[-1]:
        return x.matmul(weight)
    return x.matmul(weight.transpose(0, 1))


def _lora_linear(
    x: torch.Tensor,
    lora_a: torch.Tensor,
    lora_b: torch.Tensor,
) -> torch.Tensor:
    return x.matmul(lora_a.transpose(0, 1)).matmul(lora_b.transpose(0, 1))


def _ascend_slow_lora_forward(
    self,
    hidden_states: torch.Tensor,
    router_logits: torch.Tensor,
):
    # Startup profiling uses a large dummy batch only for memory estimation.
    # Keep that path on native Ascend fused MoE.
    if hidden_states.shape[0] > 2048:
        return self.base_layer.forward(
            hidden_states=hidden_states,
            router_logits=router_logits,
        )

    scores = torch.softmax(router_logits, dim=-1)
    topk_weights, topk_ids = torch.topk(scores, self.base_layer.top_k, dim=-1)
    if self.base_layer.renormalize:
        topk_weights = topk_weights / topk_weights.sum(dim=-1, keepdim=True)
    topk_weights = topk_weights.to(hidden_states.dtype)

    lora_index = self._active_lora_index()
    w13 = self.base_layer.w13_weight
    w2 = self.base_layer.w2_weight
    hidden_size = hidden_states.shape[-1]
    output = torch.zeros_like(hidden_states)

    for token_idx in range(hidden_states.shape[0]):
        token = hidden_states[token_idx : token_idx + 1]
        token_out = torch.zeros(
            (1, hidden_size),
            dtype=hidden_states.dtype,
            device=hidden_states.device,
        )

        for top_idx in range(topk_ids.shape[1]):
            expert_id = int(topk_ids[token_idx, top_idx].item())
            if expert_id < 0 or expert_id >= self.base_layer.local_num_experts:
                continue

            gate_up = self._expert_linear(token, w13[expert_id])

            if lora_index >= 0:
                gate_up = gate_up.clone()
                gate_size = self.w13_lora_b_stacked[0].shape[-2]
                gate_up[:, :gate_size] += self._lora_linear(
                    token,
                    self.w13_lora_a_stacked[0][lora_index, expert_id],
                    self.w13_lora_b_stacked[0][lora_index, expert_id],
                )
                if self._w13_slices == 2:
                    gate_up[:, gate_size : gate_size * 2] += self._lora_linear(
                        token,
                        self.w13_lora_a_stacked[1][lora_index, expert_id],
                        self.w13_lora_b_stacked[1][lora_index, expert_id],
                    )

            if self._w13_slices == 2:
                gate, up = gate_up.chunk(2, dim=-1)
                intermediate = torch.nn.functional.silu(gate) * up
            else:
                intermediate = torch.nn.functional.silu(gate_up)

            expert_out = self._expert_linear(intermediate, w2[expert_id])
            if lora_index >= 0:
                expert_out = expert_out + self._lora_linear(
                    intermediate,
                    self.w2_lora_a_stacked[0][lora_index, expert_id],
                    self.w2_lora_b_stacked[0][lora_index, expert_id],
                )

            token_out += expert_out * topk_weights[token_idx, top_idx]

        output[token_idx : token_idx + 1] = token_out

    shared_experts = getattr(self.base_layer, "_shared_experts", None)
    if shared_experts is not None:
        return shared_experts(hidden_states), output
    return None, output


def patch_fused_moe_lora_for_ascend() -> None:
    from vllm.lora.layers.fused_moe import FusedMoEWithLoRA

    if getattr(FusedMoEWithLoRA, "_ascend_lora_patch_applied", False):
        return

    original_init = FusedMoEWithLoRA.__init__
    original_inject = FusedMoEWithLoRA._inject_lora_into_fused_moe
    original_forward = FusedMoEWithLoRA.forward

    def patched_init(self, base_layer):
        self._use_ascend_slow_lora = True
        original_init(self, base_layer)

    def patched_inject(self):
        return None

    def patched_forward(self, *args, **kwargs):
        return self._ascend_slow_lora_forward(
            kwargs["hidden_states"],
            kwargs["router_logits"],
        )

    FusedMoEWithLoRA.__init__ = patched_init
    FusedMoEWithLoRA._inject_lora_into_fused_moe = patched_inject
    FusedMoEWithLoRA.forward = patched_forward
    FusedMoEWithLoRA._active_lora_index = _active_lora_index
    FusedMoEWithLoRA._expert_linear = staticmethod(_expert_linear)
    FusedMoEWithLoRA._lora_linear = staticmethod(_lora_linear)
    FusedMoEWithLoRA._ascend_slow_lora_forward = _ascend_slow_lora_forward
    FusedMoEWithLoRA._ascend_lora_patch_applied = True
