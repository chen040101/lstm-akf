from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence


@dataclass
class ArmorXYModelConfig:
    input_size: int = 2
    hidden_size: int = 64
    num_layers: int = 2
    dropout: float = 0.1
    max_history: int = 15
    min_history: int = 5
    future_steps: int = 15
    use_baseline: bool = True


class ResidualPredictionHead(nn.Module):
    def __init__(self, hidden_size: int, future_steps: int, dropout: float) -> None:
        super().__init__()
        self.future_steps = future_steps
        self.network = nn.Sequential(
            nn.Linear(hidden_size, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, future_steps * 2),
        )

    def forward(self, hidden_state: torch.Tensor) -> torch.Tensor:
        output = self.network(hidden_state)
        return output.view(hidden_state.shape[0], self.future_steps, 2)


class FusionGateHead(nn.Module):
    def __init__(self, hidden_size: int, dropout: float) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(hidden_size + 4, hidden_size),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, 1),
        )

    def forward(
        self,
        history_embedding: torch.Tensor,
        residual_pred_xy: torch.Tensor,
        direct_pred_xy: torch.Tensor,
    ) -> torch.Tensor:
        repeated_embedding = history_embedding.unsqueeze(1).expand(-1, residual_pred_xy.shape[1], -1)
        gate_input = torch.cat([repeated_embedding, residual_pred_xy, direct_pred_xy], dim=-1)
        return torch.sigmoid(self.network(gate_input))


class ArmorXYResidualPredictor(nn.Module):
    def __init__(self, config: Optional[ArmorXYModelConfig] = None) -> None:
        super().__init__()
        self.config = config or ArmorXYModelConfig()
        dropout = self.config.dropout if self.config.num_layers > 1 else 0.0
        self.encoder = nn.LSTM(
            input_size=self.config.input_size,
            hidden_size=self.config.hidden_size,
            num_layers=self.config.num_layers,
            dropout=dropout,
            batch_first=True,
        )
        self.head = ResidualPredictionHead(
            hidden_size=self.config.hidden_size,
            future_steps=self.config.future_steps,
            dropout=self.config.dropout,
        )
        self.direct_head = ResidualPredictionHead(
            hidden_size=self.config.hidden_size,
            future_steps=self.config.future_steps,
            dropout=self.config.dropout,
        )
        self.gate_head = FusionGateHead(
            hidden_size=self.config.hidden_size,
            dropout=self.config.dropout,
        )

    def encode_history(
        self,
        history_xy: torch.Tensor,
        history_len: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if history_len is None:
            _, (hidden_state, _) = self.encoder(history_xy)
            return hidden_state[-1]

        lengths = history_len.to(dtype=torch.long).cpu()
        max_length = int(lengths.max().item())
        trimmed_sequences = []
        for sample, length in zip(history_xy, lengths.tolist()):
            valid = sample[-length:]
            if length < max_length:
                padding = sample.new_zeros((max_length - length, sample.shape[-1]))
                valid = torch.cat([valid, padding], dim=0)
            trimmed_sequences.append(valid)
        packed_input = torch.stack(trimmed_sequences, dim=0)
        packed = pack_padded_sequence(
            packed_input,
            lengths=lengths,
            batch_first=True,
            enforce_sorted=False,
        )
        _, (hidden_state, _) = self.encoder(packed)
        return hidden_state[-1]

    def forward(
        self,
        history_xy: torch.Tensor,
        baseline_xy: Optional[torch.Tensor] = None,
        history_len: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        history_embedding = self.encode_history(history_xy, history_len=history_len)
        pred_delta_xy = self.head(history_embedding)
        direct_offset_xy = self.direct_head(history_embedding)
        last_history_xy = history_xy[:, -1, :].unsqueeze(1)
        direct_pred_xy = last_history_xy + direct_offset_xy

        if baseline_xy is None:
            baseline_xy = torch.zeros_like(pred_delta_xy)
        residual_pred_xy = baseline_xy + pred_delta_xy

        if self.config.use_baseline:
            fusion_gate = self.gate_head(history_embedding, residual_pred_xy, direct_pred_xy)
            pred_xy = (fusion_gate * residual_pred_xy) + ((1.0 - fusion_gate) * direct_pred_xy)
        else:
            fusion_gate = torch.zeros(
                (history_xy.shape[0], self.config.future_steps, 1),
                dtype=history_xy.dtype,
                device=history_xy.device,
            )
            pred_xy = direct_pred_xy
        return {
            "history_embedding": history_embedding,
            "pred_delta_xy": pred_delta_xy,
            "direct_offset_xy": direct_offset_xy,
            "baseline_xy": baseline_xy,
            "residual_pred_xy": residual_pred_xy,
            "direct_pred_xy": direct_pred_xy,
            "fusion_gate": fusion_gate,
            "pred_xy": pred_xy,
        }

    @torch.no_grad()
    def predict(
        self,
        history_xy: torch.Tensor,
        baseline_xy: Optional[torch.Tensor] = None,
        history_len: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        self.eval()
        return self.forward(history_xy, baseline_xy=baseline_xy, history_len=history_len)["pred_xy"]

    def get_model_info(self) -> Dict[str, int | bool]:
        total_params = sum(parameter.numel() for parameter in self.parameters())
        trainable_params = sum(parameter.numel() for parameter in self.parameters() if parameter.requires_grad)
        return {
            "input_size": self.config.input_size,
            "hidden_size": self.config.hidden_size,
            "num_layers": self.config.num_layers,
            "future_steps": self.config.future_steps,
            "use_baseline": self.config.use_baseline,
            "total_params": total_params,
            "trainable_params": trainable_params,
        }


class MultiStepSmoothL1Loss(nn.Module):
    def __init__(self, beta: float = 1.0) -> None:
        super().__init__()
        self.loss = nn.SmoothL1Loss(beta=beta, reduction="none")

    def forward(self, pred_xy: torch.Tensor, target_xy: torch.Tensor) -> torch.Tensor:
        return self.loss(pred_xy, target_xy).mean()


def compute_model_loss(
    outputs: Dict[str, torch.Tensor],
    target_xy: torch.Tensor,
    loss_fn: nn.Module,
    residual_aux_weight: float = 0.0,
    direct_aux_weight: float = 0.0,
) -> torch.Tensor:
    total_loss = loss_fn(outputs["pred_xy"], target_xy)
    if residual_aux_weight > 0.0:
        total_loss = total_loss + (float(residual_aux_weight) * loss_fn(outputs["residual_pred_xy"], target_xy))
    if direct_aux_weight > 0.0:
        total_loss = total_loss + (float(direct_aux_weight) * loss_fn(outputs["direct_pred_xy"], target_xy))
    return total_loss


def smoke_test() -> None:
    torch.manual_seed(0)
    model = ArmorXYResidualPredictor()
    history_xy = torch.rand(2, model.config.max_history, model.config.input_size)
    history_len = torch.tensor([model.config.max_history, model.config.min_history], dtype=torch.long)
    output = model(history_xy=history_xy, history_len=history_len)
    print({key: tuple(value.shape) for key, value in output.items()})


__all__ = [
    "ArmorXYModelConfig",
    "ArmorXYResidualPredictor",
    "FusionGateHead",
    "MultiStepSmoothL1Loss",
    "compute_model_loss",
    "ResidualPredictionHead",
    "smoke_test",
]
