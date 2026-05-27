from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import torch
import torch.nn.functional as F


@dataclass
class VecDPOLossConfig:

    beta: float = 0.1
    label_smoothing: float = 0.0
    loss_type: str = "sigmoid"
    use_evidence_weight: bool = True
    normalize_weighted_loss: bool = True
    min_evidence_weight: float = 0.5
    max_evidence_weight: float = 2.0
    reference_free: bool = False


def _ensure_1d_tensor(x: torch.Tensor, name: str) -> torch.Tensor:
    """
    Ensure input log-probability tensor is shape [batch].
    """
    if not isinstance(x, torch.Tensor):
        raise TypeError(f"{name} must be a torch.Tensor, got {type(x)}.")

    if x.dim() == 0:
        x = x.unsqueeze(0)

    if x.dim() > 1:
        x = x.view(x.shape[0], -1).sum(dim=-1)

    return x


def _prepare_evidence_weight(
    evidence_weight: Optional[torch.Tensor],
    losses: torch.Tensor,
    config: VecDPOLossConfig,
) -> torch.Tensor:
    """
    Prepare evidence weights for weighted loss computation.
    """
    if evidence_weight is None or not config.use_evidence_weight:
        return torch.ones_like(losses)

    if not isinstance(evidence_weight, torch.Tensor):
        evidence_weight = torch.tensor(
            evidence_weight,
            dtype=losses.dtype,
            device=losses.device,
        )

    evidence_weight = evidence_weight.to(device=losses.device, dtype=losses.dtype)

    if evidence_weight.dim() == 0:
        evidence_weight = evidence_weight.expand_as(losses)

    evidence_weight = evidence_weight.view(-1)

    if evidence_weight.shape[0] != losses.shape[0]:
        raise ValueError(
            f"evidence_weight batch size mismatch: "
            f"got {evidence_weight.shape[0]}, expected {losses.shape[0]}"
        )

    evidence_weight = torch.clamp(
        evidence_weight,
        min=config.min_evidence_weight,
        max=config.max_evidence_weight,
    )

    return evidence_weight


def dpo_preference_logits(
    policy_chosen_logps: torch.Tensor,
    policy_rejected_logps: torch.Tensor,
    ref_chosen_logps: Optional[torch.Tensor] = None,
    ref_rejected_logps: Optional[torch.Tensor] = None,
    reference_free: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:

    policy_chosen_logps = _ensure_1d_tensor(policy_chosen_logps, "policy_chosen_logps")
    policy_rejected_logps = _ensure_1d_tensor(policy_rejected_logps, "policy_rejected_logps")

    policy_logratios = policy_chosen_logps - policy_rejected_logps

    if reference_free:
        ref_logratios = torch.zeros_like(policy_logratios)
    else:
        if ref_chosen_logps is None or ref_rejected_logps is None:
            raise ValueError(
                "ref_chosen_logps and ref_rejected_logps are required "
                "when reference_free=False."
            )

        ref_chosen_logps = _ensure_1d_tensor(ref_chosen_logps, "ref_chosen_logps")
        ref_rejected_logps = _ensure_1d_tensor(ref_rejected_logps, "ref_rejected_logps")
        ref_logratios = ref_chosen_logps - ref_rejected_logps

    logits = policy_logratios - ref_logratios
    return logits, policy_logratios, ref_logratios


def dpo_losses_from_logits(
    logits: torch.Tensor,
    beta: float,
    loss_type: str = "sigmoid",
    label_smoothing: float = 0.0,
) -> torch.Tensor:
    """
    Compute per-sample DPO-style losses from preference logits.
    """
    if loss_type == "sigmoid":
        # Standard DPO:
        # -log sigma(beta * logits)
        # Optional conservative label smoothing:
        # -(1-eps) log sigma(beta logits) - eps log sigma(-beta logits)
        losses = (
            -F.logsigmoid(beta * logits) * (1.0 - label_smoothing)
            -F.logsigmoid(-beta * logits) * label_smoothing
        )

    elif loss_type == "hinge":
        # Hinge variant:
        # max(0, 1 - beta * logits)
        losses = torch.relu(1.0 - beta * logits)

    elif loss_type == "ipo":
        # IPO-style objective:
        # (logits - 1/(2 beta))^2
        losses = (logits - 1.0 / (2.0 * beta)) ** 2

    else:
        raise ValueError(
            f"Unsupported loss_type={loss_type}. "
            "Choose from: sigmoid, hinge, ipo."
        )

    return losses


def reduce_weighted_losses(
    losses: torch.Tensor,
    evidence_weight: Optional[torch.Tensor],
    config: VecDPOLossConfig,
) -> Tuple[torch.Tensor, torch.Tensor]:

    weights = _prepare_evidence_weight(evidence_weight, losses, config)

    if config.use_evidence_weight:
        if config.normalize_weighted_loss:
            loss = (losses * weights).sum() / weights.sum().clamp_min(1e-6)
        else:
            loss = (losses * weights).mean()
    else:
        loss = losses.mean()

    return loss, weights


def vec_dpo_loss(
    policy_chosen_logps: torch.Tensor,
    policy_rejected_logps: torch.Tensor,
    ref_chosen_logps: Optional[torch.Tensor] = None,
    ref_rejected_logps: Optional[torch.Tensor] = None,
    evidence_weight: Optional[torch.Tensor] = None,
    evidence_gap: Optional[torch.Tensor] = None,
    config: Optional[VecDPOLossConfig] = None,
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    
    config = config or VecDPOLossConfig()

    logits, policy_logratios, ref_logratios = dpo_preference_logits(
        policy_chosen_logps=policy_chosen_logps,
        policy_rejected_logps=policy_rejected_logps,
        ref_chosen_logps=ref_chosen_logps,
        ref_rejected_logps=ref_rejected_logps,
        reference_free=config.reference_free,
    )

    per_sample_losses = dpo_losses_from_logits(
        logits=logits,
        beta=config.beta,
        loss_type=config.loss_type,
        label_smoothing=config.label_smoothing,
    )

    loss, weights = reduce_weighted_losses(
        losses=per_sample_losses,
        evidence_weight=evidence_weight,
        config=config,
    )

    chosen_rewards = config.beta * (
        _ensure_1d_tensor(policy_chosen_logps, "policy_chosen_logps")
        - (
            torch.zeros_like(_ensure_1d_tensor(policy_chosen_logps, "policy_chosen_logps"))
            if config.reference_free
            else _ensure_1d_tensor(ref_chosen_logps, "ref_chosen_logps")
        )
    )

    rejected_rewards = config.beta * (
        _ensure_1d_tensor(policy_rejected_logps, "policy_rejected_logps")
        - (
            torch.zeros_like(_ensure_1d_tensor(policy_rejected_logps, "policy_rejected_logps"))
            if config.reference_free
            else _ensure_1d_tensor(ref_rejected_logps, "ref_rejected_logps")
        )
    )

    reward_margin = chosen_rewards - rejected_rewards
    reward_accuracy = (reward_margin > 0).float()

    metrics: Dict[str, torch.Tensor] = {
        "loss": loss.detach(),
        "dpo_loss_unweighted": per_sample_losses.mean().detach(),
        "dpo_loss_weighted": loss.detach(),
        "preference_logits_mean": logits.mean().detach(),
        "preference_logits_std": logits.std(unbiased=False).detach(),
        "policy_logratios_mean": policy_logratios.mean().detach(),
        "ref_logratios_mean": ref_logratios.mean().detach(),
        "chosen_rewards_mean": chosen_rewards.mean().detach(),
        "rejected_rewards_mean": rejected_rewards.mean().detach(),
        "reward_margin_mean": reward_margin.mean().detach(),
        "reward_accuracy": reward_accuracy.mean().detach(),
        "evidence_weight_mean": weights.mean().detach(),
        "evidence_weight_min": weights.min().detach(),
        "evidence_weight_max": weights.max().detach(),
    }

    if evidence_gap is not None:
        if not isinstance(evidence_gap, torch.Tensor):
            evidence_gap = torch.tensor(
                evidence_gap,
                dtype=logits.dtype,
                device=logits.device,
            )
        evidence_gap = evidence_gap.to(device=logits.device, dtype=logits.dtype).view(-1)
        metrics["evidence_gap_mean"] = evidence_gap.mean().detach()
        metrics["evidence_gap_min"] = evidence_gap.min().detach()
        metrics["evidence_gap_max"] = evidence_gap.max().detach()

    return loss, metrics


class VecDPOLoss(torch.nn.Module):
    def __init__(self, config: Optional[VecDPOLossConfig] = None):
        super().__init__()
        self.config = config or VecDPOLossConfig()

    def forward(
        self,
        policy_chosen_logps: torch.Tensor,
        policy_rejected_logps: torch.Tensor,
        ref_chosen_logps: Optional[torch.Tensor] = None,
        ref_rejected_logps: Optional[torch.Tensor] = None,
        evidence_weight: Optional[torch.Tensor] = None,
        evidence_gap: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        return vec_dpo_loss(
            policy_chosen_logps=policy_chosen_logps,
            policy_rejected_logps=policy_rejected_logps,
            ref_chosen_logps=ref_chosen_logps,
            ref_rejected_logps=ref_rejected_logps,
            evidence_weight=evidence_weight,
            evidence_gap=evidence_gap,
            config=self.config,
        )