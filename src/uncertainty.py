"""
Bayesian Uncertainty Estimation via Monte Carlo Dropout for Tactile Surfaces.

Why Uncertainty Quantification is Essential for WATER Robotics:
1. Actuator Safety: Robotic adaptive beds adjust pneumatic bladders under vulnerable bony prominences.
   Actuating on an overconfident, hallucinated landmark estimate risks aggravating pressure sores.
2. Limb Self-Occlusion: In lateral and fetal postures, contralateral limbs (e.g., right leg in left lateral)
   float above the mattress or exert negligible contact pressure. The model must signal high epistemic
   uncertainty for these non-contact keypoints rather than false certainty.
3. Postural Transition Ambiguity: Boundary frames during movement or ambiguous postures exhibit high
   predictive entropy, triggering surface stabilization rather than erratic real-time adjustments.

Mathematical Formulations:
- Predictive Distribution: p(y | x) ~ (1/T) sum_{t=1}^T p(y | x, W^(t))
- Total Predictive Entropy: H[y | x] = - sum_c p_c log2(p_c)
- Epistemic Uncertainty (Mutual Information): I[y, W | x] = H[y | x] - E_{q(W)}[H[y | x, W]]
- Landmark Spatial Uncertainty: sigma_k = sqrt(Var(x_k) + Var(y_k))
"""

from typing import Dict, List, Optional, Tuple
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.data_generator import LANDMARK_NAMES, POSTURE_NAMES


def enable_mc_dropout(model: nn.Module) -> None:
    """
    Sets Dropout modules to train mode for Monte Carlo sampling while preserving
    BatchNorm/LayerNorm in evaluation mode (freezing running mean and variance).
    """
    model.eval()
    for m in model.modules():
        if isinstance(m, (nn.Dropout, nn.Dropout2d)):
            m.train()


class BayesianTactilePredictor:
    """
    Monte Carlo Dropout Inference Engine for posture classification and landmark regression.
    """

    def __init__(
        self,
        classifier: nn.Module,
        regressor: nn.Module,
        num_mc_samples: int = 30,
        device: str = "cpu"
    ):
        self.classifier = classifier.to(device)
        self.regressor = regressor.to(device)
        self.num_mc_samples = num_mc_samples
        self.device = torch.device(device)

    def predict_posture(
        self,
        pressure_tensor: torch.Tensor
    ) -> Dict[str, np.ndarray]:
        """
        Runs T forward passes with active dropout to quantify predictive entropy and mutual information.
        
        Args:
            pressure_tensor: (B, 1, 64, 32) tensor
            
        Returns:
            Dictionary containing:
                - 'predicted_class': (B,) most likely posture indices
                - 'mean_probabilities': (B, num_classes) averaged posterior probabilities
                - 'predictive_entropy': (B,) total predictive uncertainty in bits
                - 'mutual_information': (B,) epistemic model uncertainty in bits
                - 'confidence': (B,) maximum mean probability
        """
        enable_mc_dropout(self.classifier)
        x = pressure_tensor.to(self.device)
        B = x.shape[0]

        probs_samples = []
        with torch.no_grad():
            for _ in range(self.num_mc_samples):
                logits = self.classifier(x)
                probs = F.softmax(logits, dim=-1).cpu().numpy()  # (B, 6)
                probs_samples.append(probs)

        # Shape: (T, B, 6)
        probs_samples = np.array(probs_samples)
        # Mean probability: (B, 6)
        mean_probs = np.mean(probs_samples, axis=0)
        predicted_classes = np.argmax(mean_probs, axis=-1)
        confidences = np.max(mean_probs, axis=-1)

        # 1. Total Predictive Entropy: H[y|x] = - sum_c p_bar_c * log2(p_bar_c)
        eps = 1e-8
        total_entropy = -np.sum(mean_probs * np.log2(mean_probs + eps), axis=-1)

        # 2. Expected Entropy across samples (Aleatoric): E_q[H[y|x, W]]
        sample_entropies = -np.sum(probs_samples * np.log2(probs_samples + eps), axis=-1)
        expected_entropy = np.mean(sample_entropies, axis=0)

        # 3. Mutual Information (Epistemic): I[y, W|x] = Total - Expected
        mutual_info = np.maximum(total_entropy - expected_entropy, 0.0)

        return {
            "predicted_class": predicted_classes,
            "mean_probabilities": mean_probs,
            "predictive_entropy": total_entropy,
            "mutual_information": mutual_info,
            "confidence": confidences
        }

    def predict_landmarks(
        self,
        pressure_tensor: torch.Tensor,
        posture_idx: torch.Tensor
    ) -> Dict[str, np.ndarray]:
        """
        Runs T forward passes with active dropout to quantify spatial landmark uncertainty.
        
        Args:
            pressure_tensor: (B, 1, 64, 32) tensor
            posture_idx: (B,) posture class indices
            
        Returns:
            Dictionary containing:
                - 'mean_coords_px': (B, 9, 2) predictive mean landmark coordinates in pixels
                - 'std_coords_px': (B, 9, 2) standard deviation along (x, y)
                - 'spatial_uncertainty': (B, 9) combined Euclidean standard deviation sigma_k
                - 'covariances': (B, 9, 2, 2) spatial covariance matrices per landmark
        """
        enable_mc_dropout(self.regressor)
        x = pressure_tensor.to(self.device)
        postures = posture_idx.to(self.device)
        B = x.shape[0]

        coords_samples = []
        with torch.no_grad():
            for _ in range(self.num_mc_samples):
                coords = self.regressor(x, postures).cpu().numpy()  # (B, 9, 2)
                coords_samples.append(coords)

        # Shape: (T, B, 9, 2)
        coords_samples = np.array(coords_samples)
        # Predictive mean coordinate: (B, 9, 2)
        mean_coords = np.mean(coords_samples, axis=0)
        # Coordinate standard deviations: (B, 9, 2)
        std_coords = np.std(coords_samples, axis=0)

        # Euclidean spatial uncertainty: sigma_k = sqrt(sigma_x^2 + sigma_y^2)
        spatial_uncertainty = np.sqrt(np.sum(std_coords ** 2, axis=-1))  # (B, 9)

        # Covariance matrices: (B, 9, 2, 2)
        diff = coords_samples - mean_coords[np.newaxis, ...]  # (T, B, 9, 2)
        # cov[b, k, i, j] = (1/T) sum_t diff[t, b, k, i] * diff[t, b, k, j]
        covariances = np.mean(
            diff[..., :, np.newaxis] * diff[..., np.newaxis, :],
            axis=0
        )  # (B, 9, 2, 2)

        return {
            "mean_coords_px": mean_coords,
            "std_coords_px": std_coords,
            "spatial_uncertainty": spatial_uncertainty,
            "covariances": covariances
        }


def analyze_occlusion_uncertainty(
    spatial_uncertainty: np.ndarray,
    posture_indices: np.ndarray,
    landmark_names: Optional[List[str]] = None
) -> Dict[str, Dict[str, float]]:
    """
    Analyzes landmark uncertainty across postures, highlighting elevated uncertainty
    for self-occluded anatomical landmarks (e.g. contralateral limbs in lateral postures).
    
    Args:
        spatial_uncertainty: (N, 9) landmark spatial standard deviations in pixels
        posture_indices: (N,) posture class labels
        landmark_names: Optional list of 9 landmark names
        
    Returns:
        Structured breakdown: posture -> landmark_name -> average uncertainty (px)
    """
    if landmark_names is None:
        landmark_names = LANDMARK_NAMES

    report: Dict[str, Dict[str, float]] = {}

    for p_idx, p_name in enumerate(POSTURE_NAMES):
        mask = (posture_indices == p_idx)
        if not np.any(mask):
            continue
        p_unc = spatial_uncertainty[mask]  # (N_p, 9)
        mean_per_lm = np.mean(p_unc, axis=0)
        report[p_name] = {
            landmark_names[k]: float(mean_per_lm[k])
            for k in range(len(landmark_names))
        }

    return report


if __name__ == "__main__":
    from src.models import PostureClassifier, LandmarkRegressor
    clf = PostureClassifier()
    reg = LandmarkRegressor()
    predictor = BayesianTactilePredictor(clf, reg, num_mc_samples=5)
    dummy_x = torch.randn(2, 1, 64, 32)
    p_res = predictor.predict_posture(dummy_x)
    l_res = predictor.predict_landmarks(dummy_x, torch.tensor([0, 1]))
    print("Bayesian uncertainty test passed!")
    print(f"Predictive entropy: {p_res['predictive_entropy']}")
    print(f"Landmark spatial uncertainty shape: {l_res['spatial_uncertainty'].shape}")
