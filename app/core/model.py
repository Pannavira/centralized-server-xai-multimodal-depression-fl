"""
app/core/model.py
=================
Arsitektur Multimodal Deep Learning Baseline (FED-MIND)
Mencakup Audio, Visual, Text Encoders + Late Fusion Regressor.
"""

from typing import Any, Dict, Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F


class TemporalAttentionPooling(nn.Module):
    """Self-Attention Pooling untuk merangkum representasi temporal BiLSTM."""

    def __init__(self, in_features: int) -> None:
        super().__init__()
        self.attn = nn.Sequential(
            nn.Linear(in_features, in_features // 2),
            nn.Tanh(),
            nn.Linear(in_features // 2, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, D)
        weights = self.attn(x)
        weights = F.softmax(weights, dim=1)
        pooled = torch.sum(x * weights, dim=1)
        return pooled

    def get_attention_weights(self, x: torch.Tensor) -> torch.Tensor:
        """Mengembalikan peta bobot perhatian temporal (B, T) untuk analisis XAI."""
        weights = self.attn(x)
        weights = F.softmax(weights, dim=1).squeeze(-1)
        return weights


class AudioEncoder(nn.Module):
    """
    Audio Feature Extractor:
    Input: (B, 512, 80) log-Mel spectrogram.
    Conv1d(80->128, k=3, p=1) -> BatchNorm1d(128) -> ReLU -> MaxPool1d(2)
    -> Dropout -> BiLSTM(128, 256, num_layers=2) -> Pooling -> Output: (B, 256)
    """

    def __init__(
        self,
        in_channels: int = 80,
        conv_out_channels: int = 128,
        lstm_hidden_size: int = 256,
        lstm_num_layers: int = 2,
        dropout_conv: float = 0.4,
        output_dim: int = 256,
        pooling_type: str = "mean",
    ) -> None:
        super().__init__()
        self.pooling_type = pooling_type
        self.conv = nn.Sequential(
            nn.Conv1d(in_channels=in_channels, out_channels=conv_out_channels, kernel_size=3, padding=1),
            nn.BatchNorm1d(conv_out_channels),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=2),
            nn.Dropout(p=dropout_conv),
        )

        self.lstm = nn.LSTM(
            input_size=conv_out_channels,
            hidden_size=lstm_hidden_size,
            num_layers=lstm_num_layers,
            batch_first=True,
            bidirectional=True,
        )

        if pooling_type == "attention":
            self.pool = TemporalAttentionPooling(lstm_hidden_size * 2)
        else:
            self.pool = None

        self.fc = nn.Linear(lstm_hidden_size * 2, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Expected input: (B, 512, 80) -> transpose to (B, 80, 512) for Conv1D
        if x.dim() == 3 and x.size(1) == 512 and x.size(2) == 80:
            x = x.permute(0, 2, 1)

        h = self.conv(x)
        h = h.permute(0, 2, 1)  # (B, 256, conv_out_channels)
        lstm_out, _ = self.lstm(h)

        if self.pool is not None:
            h_pool = self.pool(lstm_out)
        else:
            h_pool = torch.mean(lstm_out, dim=1)

        out = self.fc(h_pool)
        return out


class VisualEncoder(nn.Module):
    """
    Visual Feature Extractor (3D Facial Landmarks + Gaze):
    Input: (B, 512, 72, 3) koordinat dinamika wajah.
    Conv2d(3->64, k=(3, 72), p=(1, 0)) -> BatchNorm1d(64) -> ReLU
    -> MaxPool1d(2) -> Dropout -> BiLSTM(64, 256, num_layers=2) -> Pooling -> Output: (B, 256)
    """

    def __init__(
        self,
        in_channels: int = 3,
        num_nodes: int = 72,
        conv_out_channels: int = 64,
        lstm_hidden_size: int = 256,
        lstm_num_layers: int = 2,
        dropout_conv: float = 0.4,
        output_dim: int = 256,
        pooling_type: str = "mean",
    ) -> None:
        super().__init__()
        self.pooling_type = pooling_type
        self.conv2d = nn.Conv2d(
            in_channels=in_channels,
            out_channels=conv_out_channels,
            kernel_size=(3, num_nodes),
            padding=(1, 0),
        )

        self.temporal_block = nn.Sequential(
            nn.BatchNorm1d(conv_out_channels),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=2),
            nn.Dropout(p=dropout_conv),
        )

        self.lstm = nn.LSTM(
            input_size=conv_out_channels,
            hidden_size=lstm_hidden_size,
            num_layers=lstm_num_layers,
            batch_first=True,
            bidirectional=True,
        )

        if pooling_type == "attention":
            self.pool = TemporalAttentionPooling(lstm_hidden_size * 2)
        else:
            self.pool = None

        self.fc = nn.Linear(lstm_hidden_size * 2, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Expected input: (B, 512, 72, 3) -> transpose to (B, 3, 512, 72)
        if x.dim() == 4 and x.size(1) == 512 and x.size(2) == 72 and x.size(3) == 3:
            x = x.permute(0, 3, 1, 2)

        h = self.conv2d(x)
        h = h.squeeze(-1)  # (B, conv_out_channels, 512)
        h = self.temporal_block(h)
        h = h.permute(0, 2, 1)  # (B, 256, conv_out_channels)
        lstm_out, _ = self.lstm(h)

        if self.pool is not None:
            h_pool = self.pool(lstm_out)
        else:
            h_pool = torch.mean(lstm_out, dim=1)

        out = self.fc(h_pool)
        return out


class TextEncoder(nn.Module):
    """
    Text Feature Extractor (Sentence-BERT Embeddings):
    Input: (B, 512, 768) embeddings semantik kalimat.
    Conv1d(768->128, k=3, p=1) -> BatchNorm1d(128) -> ReLU -> MaxPool1d(2)
    -> Dropout -> BiLSTM(128, 256, num_layers=2) -> Pooling -> Output: (B, 256)
    """

    def __init__(
        self,
        in_channels: int = 768,
        conv_out_channels: int = 128,
        lstm_hidden_size: int = 256,
        lstm_num_layers: int = 2,
        dropout_conv: float = 0.4,
        output_dim: int = 256,
        pooling_type: str = "mean",
    ) -> None:
        super().__init__()
        self.pooling_type = pooling_type
        self.conv = nn.Sequential(
            nn.Conv1d(in_channels=in_channels, out_channels=conv_out_channels, kernel_size=3, padding=1),
            nn.BatchNorm1d(conv_out_channels),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=2),
            nn.Dropout(p=dropout_conv),
        )

        self.lstm = nn.LSTM(
            input_size=conv_out_channels,
            hidden_size=lstm_hidden_size,
            num_layers=lstm_num_layers,
            batch_first=True,
            bidirectional=True,
        )

        if pooling_type == "attention":
            self.pool = TemporalAttentionPooling(lstm_hidden_size * 2)
        else:
            self.pool = None

        self.fc = nn.Linear(lstm_hidden_size * 2, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Expected input: (B, 512, 768) -> transpose to (B, 768, 512)
        if x.dim() == 3 and x.size(1) == 512 and x.size(2) == 768:
            x = x.permute(0, 2, 1)

        h = self.conv(x)
        h = h.permute(0, 2, 1)  # (B, 256, conv_out_channels)
        lstm_out, _ = self.lstm(h)

        if self.pool is not None:
            h_pool = self.pool(lstm_out)
        else:
            h_pool = torch.mean(lstm_out, dim=1)

        out = self.fc(h_pool)
        return out


class MultimodalBaselineModel(nn.Module):
    """
    Multimodal Depression Detection Baseline Model.
    Menerima Audio, Visual, dan Text -> Late Fusion -> Regresi PHQ-8.
    """

    def __init__(
        self,
        audio_in_channels: int = 80,
        visual_in_channels: int = 3,
        visual_num_nodes: int = 72,
        text_in_channels: int = 768,
        unimodal_output_dim: int = 256,
        regressor_hidden_dims: Tuple[int, ...] = (128,),
        dropout_conv: float = 0.4,
        regressor_dropout: float = 0.3,
        pooling_type: str = "mean",
    ) -> None:
        super().__init__()
        self.audio_encoder = AudioEncoder(
            in_channels=audio_in_channels,
            output_dim=unimodal_output_dim,
            dropout_conv=dropout_conv,
            pooling_type=pooling_type,
        )
        self.visual_encoder = VisualEncoder(
            in_channels=visual_in_channels,
            num_nodes=visual_num_nodes,
            output_dim=unimodal_output_dim,
            dropout_conv=dropout_conv,
            pooling_type=pooling_type,
        )
        self.text_encoder = TextEncoder(
            in_channels=text_in_channels,
            output_dim=unimodal_output_dim,
            dropout_conv=dropout_conv,
            pooling_type=pooling_type,
        )

        fusion_dim = unimodal_output_dim * 3  # 768

        layers = []
        in_d = fusion_dim
        for h_dim in regressor_hidden_dims:
            layers.append(nn.Linear(in_d, h_dim))
            layers.append(nn.ReLU(inplace=True))
            layers.append(nn.Dropout(p=regressor_dropout))
            in_d = h_dim
        layers.append(nn.Linear(in_d, 1))

        self.regressor = nn.Sequential(*layers)

    def forward(
        self,
        audio: torch.Tensor,
        visual: torch.Tensor,
        text: torch.Tensor,
    ) -> torch.Tensor:
        I_a = self.audio_encoder(audio)
        I_v = self.visual_encoder(visual)
        I_t = self.text_encoder(text)
        x_concat = torch.cat([I_a, I_v, I_t], dim=-1)
        pred_phq8 = self.regressor(x_concat)
        return pred_phq8

    def extract_features(
        self,
        audio: torch.Tensor,
        visual: torch.Tensor,
        text: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """Ekstraksi representasi laten multimodal untuk analisis XAI."""
        I_a = self.audio_encoder(audio)
        I_v = self.visual_encoder(visual)
        I_t = self.text_encoder(text)
        x_concat = torch.cat([I_a, I_v, I_t], dim=-1)
        return {
            "audio_latent": I_a,
            "visual_latent": I_v,
            "text_latent": I_t,
            "fused_latent": x_concat,
        }


def build_model_from_config(config: Dict[str, Any]) -> MultimodalBaselineModel:
    """Factory helper untuk membangun MultimodalBaselineModel dari dict konfigurasi."""
    model_cfg = config.get("model", {})
    audio_cfg = model_cfg.get("audio_encoder", {})
    visual_cfg = model_cfg.get("visual_encoder", {})
    text_cfg = model_cfg.get("text_encoder", {})

    audio_in = audio_cfg.get("in_channels", 80)
    visual_in = visual_cfg.get("in_channels", 3)
    visual_nodes = 72
    if "kernel_size" in visual_cfg and isinstance(visual_cfg["kernel_size"], (list, tuple)):
        visual_nodes = visual_cfg["kernel_size"][1]

    text_in = text_cfg.get("in_channels", 768)
    unimodal_out = audio_cfg.get("output_dim", 256)
    dropout_conv = audio_cfg.get("dropout_conv", 0.4)
    pooling_type = audio_cfg.get("pooling_type", "mean")

    reg_dims = model_cfg.get("regressor_hidden_dims", [128])
    if isinstance(reg_dims, list):
        reg_dims = tuple(reg_dims)
    reg_dropout = model_cfg.get("dropout", 0.3)

    return MultimodalBaselineModel(
        audio_in_channels=audio_in,
        visual_in_channels=visual_in,
        visual_num_nodes=visual_nodes,
        text_in_channels=text_in,
        unimodal_output_dim=unimodal_out,
        regressor_hidden_dims=reg_dims,
        dropout_conv=dropout_conv,
        regressor_dropout=reg_dropout,
        pooling_type=pooling_type,
    )
