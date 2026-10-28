"""
Deep Learning Architectures for Tactile Pressure Posture Recognition & Body Landmark Estimation.

Includes:
1. SpatialAttentionModule: Channel-compressed spatial attention mechanism highlighting
   anatomical contact points (sacrum, scapula, trochanters, heels).
2. PostureClassifier: CNN with spatial attention and Monte Carlo Dropout for 6-class
   posture recognition and predictive entropy estimation.
3. LandmarkRegressor: Posture-conditioned multi-scale CNN regressing 9 anatomical
   keypoint coordinates (18 values: x, y pairs).
4. PCK Evaluation Metric: Percentage of Correct Keypoints under varying Euclidean thresholds.
"""

from typing import Dict, Optional, Tuple, Union
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
import numpy as np


class SpatialAttentionModule(nn.Module):
    """
    Spatial Attention Module (CBAM-style spatial attention).
    
    Computes a 2D spatial attention weight map M_s in [0, 1]^(H x W) by concatenating
    channel-wise average pooling and max pooling, followed by a 7x7 convolution:
        M_s(F) = sigmoid(Conv7x7([AvgPool(F); MaxPool(F)]))
    """

    def __init__(self, kernel_size: int = 7):
        super().__init__()
        assert kernel_size in (3, 7), "Kernel size must be 3 or 7"
        padding = 3 if kernel_size == 7 else 1
        self.conv = nn.Conv2d(2, 1, kernel_size=kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # x: (B, C, H, W)
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        combined = torch.cat([avg_out, max_out], dim=1)
        scale = self.sigmoid(self.conv(combined))
        return x * scale, scale


class PostureClassifier(nn.Module):
    """
    Spatial-Attention CNN for Posture Recognition with Monte Carlo Dropout.
    
    Input: (B, 1, 64, 32) pressure array
    Output: (B, num_classes) classification logits
    """

    def __init__(
        self,
        in_channels: int = 1,
        base_filters: int = 32,
        num_classes: int = 6,
        mc_dropout_prob: float = 0.30
    ):
        super().__init__()
        self.mc_dropout_prob = mc_dropout_prob

        # Stage 1: Local tactile contact extraction (64 x 32) -> (32 x 16)
        self.stage1 = nn.Sequential(
            nn.Conv2d(in_channels, base_filters, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(base_filters),
            nn.GELU(),
            nn.Conv2d(base_filters, base_filters, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(base_filters),
            nn.GELU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        # Stage 2: Regional load configuration (32 x 16) -> (16 x 8)
        self.stage2 = nn.Sequential(
            nn.Conv2d(base_filters, base_filters * 2, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(base_filters * 2),
            nn.GELU(),
            nn.Conv2d(base_filters * 2, base_filters * 2, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(base_filters * 2),
            nn.GELU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        # Spatial Attention over regional contact features
        self.spatial_attention = SpatialAttentionModule(kernel_size=7)

        # Stage 3: Global body silhouette embedding (16 x 8) -> (8 x 4)
        self.stage3 = nn.Sequential(
            nn.Conv2d(base_filters * 2, base_filters * 4, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(base_filters * 4),
            nn.GELU(),
            nn.AdaptiveAvgPool2d((1, 1))
        )

        # Classifier head with MC Dropout
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(base_filters * 4, 64),
            nn.GELU(),
            nn.Dropout(p=mc_dropout_prob),
            nn.Linear(64, num_classes)
        )

    def forward(
        self,
        x: torch.Tensor,
        return_attention: bool = False
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        out1 = self.stage1(x)
        out2 = self.stage2(out1)
        attended, attn_weights = self.spatial_attention(out2)
        out3 = self.stage3(attended)
        logits = self.classifier(out3)

        if return_attention:
            return logits, attn_weights
        return logits


class LandmarkRegressor(nn.Module):
    """
    Posture-Conditioned Deep Keypoint Regressor for 9 Body Landmarks.
    
    Conditioning principle: Since limb visibility and contact mechanics differ dramatically
    between lateral, prone, supine, and seated postures, posture-conditioning provides a strong
    biomechanical structural prior that guides keypoint regression under occlusion.
    
    Input:
        x: (B, 1, 64, 32) pressure array
        posture_idx: (B,) posture class index [0, 5]
    Output:
        coords: (B, 9, 2) in grid pixel space [x in [0, 31], y in [0, 63]]
    """

    def __init__(
        self,
        in_channels: int = 1,
        base_filters: int = 32,
        posture_cond_dim: int = 16,
        num_landmarks: int = 9,
        grid_height: int = 64,
        grid_width: int = 32,
        mc_dropout_prob: float = 0.25
    ):
        super().__init__()
        self.num_landmarks = num_landmarks
        self.H = float(grid_height)
        self.W = float(grid_width)
        self.mc_dropout_prob = mc_dropout_prob

        # Learned categorical posture embedding
        self.posture_embedding = nn.Embedding(num_embeddings=6, embedding_dim=posture_cond_dim)

        # Deep spatial encoder
        self.encoder = nn.Sequential(
            # (1, 64, 32) -> (32, 32, 16)
            nn.Conv2d(in_channels, base_filters, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(base_filters),
            nn.GELU(),
            nn.MaxPool2d(2, 2),

            # (32, 32, 16) -> (64, 16, 8)
            nn.Conv2d(base_filters, base_filters * 2, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(base_filters * 2),
            nn.GELU(),
            nn.MaxPool2d(2, 2),

            # (64, 16, 8) -> (128, 8, 4)
            nn.Conv2d(base_filters * 2, base_filters * 4, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(base_filters * 4),
            nn.GELU(),
            nn.MaxPool2d(2, 2),
        )

        feature_dim = base_filters * 4 * 8 * 4  # 128 * 32 = 4096
        self.fc_features = nn.Sequential(
            nn.Flatten(),
            nn.Linear(feature_dim, 128),
            nn.GELU(),
            nn.Dropout(p=mc_dropout_prob)
        )

        # Multi-modal fusion layer (pressure visual features + posture prior)
        total_dim = 128 + posture_cond_dim
        self.regressor_head = nn.Sequential(
            nn.Linear(total_dim, 128),
            nn.GELU(),
            nn.Dropout(p=mc_dropout_prob),
            nn.Linear(128, 64),
            nn.GELU(),
            nn.Linear(64, num_landmarks * 2),
            nn.Sigmoid()  # Normalized coordinates in [0, 1]
        )

    def forward(
        self,
        x: torch.Tensor,
        posture_idx: torch.Tensor,
        return_normalized: bool = False
    ) -> torch.Tensor:
        """
        Forward pass.
        
        Returns:
            If return_normalized: (B, 9, 2) in [0, 1]
            Else: (B, 9, 2) in physical sensor pixels [0, W] x [0, H]
        """
        feat_map = self.encoder(x)
        feat_vec = self.fc_features(feat_map)
        posture_embed = self.posture_embedding(posture_idx)

        combined = torch.cat([feat_vec, posture_embed], dim=1)
        norm_coords = self.regressor_head(combined).view(-1, self.num_landmarks, 2)

        if return_normalized:
            return norm_coords

        # Scale to sensor pixel coordinates
        # norm_coords[:, :, 0] is x in [0, 1] -> scale by W
        # norm_coords[:, :, 1] is y in [0, 1] -> scale by H
        scale_tensor = torch.tensor([self.W, self.H], device=x.device, dtype=x.dtype)
        coords_px = norm_coords * scale_tensor
        return coords_px


class TactilePressureDataset(Dataset):
    """PyTorch Dataset wrapper for synthetic and experimental pressure mat data."""

    def __init__(
        self,
        pressure_maps: np.ndarray,
        landmarks: np.ndarray,
        postures: np.ndarray,
        subject_ids: np.ndarray,
        grid_height: int = 64,
        grid_width: int = 32
    ):
        # Shape: (N, 1, H, W)
        if pressure_maps.ndim == 3:
            pressure_maps = np.expand_dims(pressure_maps, axis=1)

        self.pressure_maps = torch.from_numpy(pressure_maps).float()
        self.landmarks = torch.from_numpy(landmarks).float()
        self.postures = torch.from_numpy(postures).long()
        self.subject_ids = torch.from_numpy(subject_ids).long()

        # Normalized ground truth coordinates in [0, 1]
        self.norm_landmarks = self.landmarks.clone()
        self.norm_landmarks[..., 0] /= float(grid_width)
        self.norm_landmarks[..., 1] /= float(grid_height)

    def __len__(self) -> int:
        return len(self.pressure_maps)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        return {
            "pressure": self.pressure_maps[idx],
            "landmarks_px": self.landmarks[idx],
            "landmarks_norm": self.norm_landmarks[idx],
            "posture": self.postures[idx],
            "subject_id": self.subject_ids[idx]
        }


def compute_pck(
    pred_coords_px: Union[np.ndarray, torch.Tensor],
    gt_coords_px: Union[np.ndarray, torch.Tensor],
    threshold_px: float = 10.0
) -> Tuple[float, np.ndarray]:
    """
    Computes Percentage of Correct Keypoints (PCK@threshold_px).
    
    A predicted landmark is correct if Euclidean distance ||p - gt||_2 <= threshold_px.
    
    Args:
        pred_coords_px: (N, 9, 2) predicted coordinates in pixels
        gt_coords_px: (N, 9, 2) ground truth coordinates in pixels
        threshold_px: Error tolerance radius in pixels (default: 10.0 px)
        
    Returns:
        pck_mean: Average PCK across all landmarks in percent [0, 100]
        pck_per_landmark: (9,) PCK for each individual landmark in percent
    """
    if isinstance(pred_coords_px, torch.Tensor):
        pred_coords_px = pred_coords_px.detach().cpu().numpy()
    if isinstance(gt_coords_px, torch.Tensor):
        gt_coords_px = gt_coords_px.detach().cpu().numpy()

    # Euclidean distance per landmark: (N, 9)
    dist = np.sqrt(np.sum((pred_coords_px - gt_coords_px) ** 2, axis=-1))
    correct = (dist <= threshold_px).astype(np.float32)

    pck_per_landmark = np.mean(correct, axis=0) * 100.0
    pck_mean = float(np.mean(correct) * 100.0)

    return pck_mean, pck_per_landmark


def compute_mean_pixel_error(
    pred_coords_px: Union[np.ndarray, torch.Tensor],
    gt_coords_px: Union[np.ndarray, torch.Tensor]
) -> Tuple[float, np.ndarray]:
    """
    Computes Mean Pixel Error (Euclidean distance in grid pixels).
    
    Returns:
        mean_error: Global average error across all frames and landmarks
        per_landmark_error: (9,) Average error per individual landmark
    """
    if isinstance(pred_coords_px, torch.Tensor):
        pred_coords_px = pred_coords_px.detach().cpu().numpy()
    if isinstance(gt_coords_px, torch.Tensor):
        gt_coords_px = gt_coords_px.detach().cpu().numpy()

    dist = np.sqrt(np.sum((pred_coords_px - gt_coords_px) ** 2, axis=-1))
    per_landmark_error = np.mean(dist, axis=0)
    mean_error = float(np.mean(dist))

    return mean_error, per_landmark_error


if __name__ == "__main__":
    clf = PostureClassifier()
    reg = LandmarkRegressor()
    dummy_x = torch.randn(4, 1, 64, 32)
    posture_lbl = torch.tensor([0, 1, 2, 3])

    logits = clf(dummy_x)
    coords = reg(dummy_x, posture_lbl)

    print(f"Classifier output shape: {logits.shape}")
    print(f"Regressor output shape: {coords.shape}")
    assert logits.shape == (4, 6)
    assert coords.shape == (4, 9, 2)
    print("Models self-test passed successfully!")
