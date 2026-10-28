"""
Publication-Quality Visualization Tools for Pressure Sensing & Biomechanical Estimation.

Key Visualizations:
1. Tactile Pressure Map with Ground Truth vs Predicted Anatomical Skeleton & Uncertainty Ellipses
2. Mattress PSF Deconvolution Comparison (Raw vs Deconvolved vs Ground Truth Contact)
3. Normalized Confusion Matrix for 6 Posture Classes
4. Per-Landmark Euclidean Error Distribution (Boxplot/Violin)
5. Uncertainty Calibration Curve (Predicted Uncertainty vs Observed Error)
6. Leave-One-Subject-Out (LOSO) Cross-Person Generalisation Summary
"""

from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import matplotlib
matplotlib.use("Agg")  # Headless backend for research environments
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import seaborn as sns

from src.data_generator import POSTURE_NAMES, LANDMARK_NAMES

# Kinematic skeleton pairs connecting anatomical keypoints:
# 0: Head, 1: L Sh, 2: R Sh, 3: L Hip, 4: R Hip, 5: L Knee, 6: R Knee, 7: L Ankle, 8: R Ankle
SKELETON_EDGES = [
    (0, 1), (0, 2),        # Head to shoulders
    (1, 2),                # Shoulder girdle
    (1, 3), (2, 4),        # Torso lateral flanks
    (3, 4),                # Pelvic girdle
    (3, 5), (5, 7),        # Left leg (Hip -> Knee -> Ankle)
    (4, 6), (6, 8)         # Right leg (Hip -> Knee -> Ankle)
]


def plot_pressure_with_landmarks(
    pressure_map: np.ndarray,
    gt_landmarks: Optional[np.ndarray] = None,
    pred_landmarks: Optional[np.ndarray] = None,
    covariances: Optional[np.ndarray] = None,
    title: str = "Tactile Pressure Array & Keypoint Estimation",
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Plots the 64x32 pressure array heatmap overlaid with Ground Truth (green)
    and Predicted (red) landmarks, connected by kinematic skeleton edges,
    plus 1-sigma uncertainty ellipses.
    
    Args:
        pressure_map: (64, 32) 2D array
        gt_landmarks: (9, 2) array of ground truth coordinates [x, y]
        pred_landmarks: (9, 2) array of predicted coordinates [x, y]
        covariances: (9, 2, 2) optional spatial covariance matrices for error ellipses
        title: Plot title string
        save_path: Optional path to save image
    """
    fig, ax = plt.subplots(figsize=(6, 10), dpi=150)
    
    # Heatmap of pressure distribution
    im = ax.imshow(
        pressure_map,
        cmap="inferno",
        origin="upper",
        aspect="equal",
        interpolation="bilinear"
    )
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Contact Pressure (Normalized / kPa)", fontsize=10)

    # Plot Ground Truth Skeleton (Green)
    if gt_landmarks is not None:
        for u, v in SKELETON_EDGES:
            ax.plot(
                [gt_landmarks[u, 0], gt_landmarks[v, 0]],
                [gt_landmarks[u, 1], gt_landmarks[v, 1]],
                color="#00FF66",
                linestyle="--",
                linewidth=1.8,
                alpha=0.85
            )
        ax.scatter(
            gt_landmarks[:, 0],
            gt_landmarks[:, 1],
            color="#00FF66",
            s=45,
            edgecolors="black",
            linewidth=0.8,
            zorder=4,
            label="Ground Truth"
        )

    # Plot Predicted Skeleton & Uncertainty Ellipses (Cyan/Red)
    if pred_landmarks is not None:
        for u, v in SKELETON_EDGES:
            ax.plot(
                [pred_landmarks[u, 0], pred_landmarks[v, 0]],
                [pred_landmarks[u, 1], pred_landmarks[v, 1]],
                color="#FF3366",
                linestyle="-",
                linewidth=2.0,
                alpha=0.9
            )
        ax.scatter(
            pred_landmarks[:, 0],
            pred_landmarks[:, 1],
            color="#FF3366",
            marker="x",
            s=55,
            linewidth=2.0,
            zorder=5,
            label="Predicted"
        )

        # Plot 1-sigma uncertainty ellipses
        if covariances is not None:
            for k in range(len(pred_landmarks)):
                cov = covariances[k]
                eigvals, eigvecs = np.linalg.eigh(cov)
                order = eigvals.argsort()[::-1]
                eigvals, eigvecs = eigvals[order], eigvecs[:, order]
                # 1 standard deviation radius
                width = 2.0 * np.sqrt(max(1e-4, eigvals[0]))
                height = 2.0 * np.sqrt(max(1e-4, eigvals[1]))
                angle = np.degrees(np.arctan2(eigvecs[1, 0], eigvecs[0, 0]))

                ellipse = patches.Ellipse(
                    xy=(pred_landmarks[k, 0], pred_landmarks[k, 1]),
                    width=width,
                    height=height,
                    angle=angle,
                    edgecolor="#FFCC00",
                    facecolor="#FFCC00",
                    alpha=0.25,
                    linewidth=1.2,
                    zorder=3
                )
                ax.add_patch(ellipse)

    ax.set_title(title, fontsize=12, fontweight="bold", pad=10)
    ax.set_xlabel("Lateral Sensel Index (x: 0..31)", fontsize=10)
    ax.set_ylabel("Cranio-Caudal Sensel Index (y: 0..63)", fontsize=10)
    ax.set_xlim(-0.5, 31.5)
    ax.set_ylim(63.5, -0.5)
    ax.legend(loc="upper right", framealpha=0.85, fontsize=9)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, bbox_inches="tight")
    return fig


def plot_deconvolution_comparison(
    raw_map: np.ndarray,
    deconvolved_map: np.ndarray,
    clean_map: np.ndarray,
    slice_row: int = 34,
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    3-panel visual proof of Wiener Deconvolution resolving the mattress inverse problem,
    plus 1D profile slice showing peak restoration at bony prominences.
    """
    fig = plt.figure(figsize=(14, 6), dpi=150)
    gs = fig.add_gridspec(2, 3, height_ratios=[3, 1.2])

    ax0 = fig.add_subplot(gs[0, 0])
    ax1 = fig.add_subplot(gs[0, 1])
    ax2 = fig.add_subplot(gs[0, 2])
    ax_profile = fig.add_subplot(gs[1, :])

    vmax = max(np.max(raw_map), np.max(deconvolved_map), np.max(clean_map))

    im0 = ax0.imshow(raw_map, cmap="inferno", vmin=0, vmax=vmax)
    ax0.set_title("1. Raw Degraded Sensel Map\n(PSF Blur + Noise + Dead Sensels)", fontsize=10, fontweight="bold")
    ax0.axhline(slice_row, color="cyan", linestyle="--", linewidth=1.2, label=f"Slice (y={slice_row})")
    ax0.legend(loc="lower left", fontsize=8)

    im1 = ax1.imshow(deconvolved_map, cmap="inferno", vmin=0, vmax=vmax)
    ax1.set_title("2. Wiener Deconvolved Map\n(Inverted Compliant Layer)", fontsize=10, fontweight="bold")
    ax1.axhline(slice_row, color="cyan", linestyle="--", linewidth=1.2)

    im2 = ax2.imshow(clean_map, cmap="inferno", vmin=0, vmax=vmax)
    ax2.set_title("3. True Interfacial Contact Map\n(Ground Truth Contact Mechanics)", fontsize=10, fontweight="bold")
    ax2.axhline(slice_row, color="cyan", linestyle="--", linewidth=1.2)

    for ax in [ax0, ax1, ax2]:
        ax.set_xlabel("Sensel x")
        ax.set_ylabel("Sensel y")

    # 1D slice comparison across pelvic/sacrum row
    xs = np.arange(raw_map.shape[1])
    ax_profile.plot(xs, raw_map[slice_row, :], "r--", label="Raw Degraded (Blurred)", linewidth=1.8)
    ax_profile.plot(xs, deconvolved_map[slice_row, :], "b-", label="Wiener Deconvolved (Sharpened)", linewidth=2.0)
    ax_profile.plot(xs, clean_map[slice_row, :], "g:", label="Ground Truth Interfacial Contact", linewidth=2.2)
    ax_profile.set_title(f"Cross-Sectional Pressure Profile Across Pelvis/Sacrum (y = {slice_row})", fontsize=10, fontweight="bold")
    ax_profile.set_xlabel("Lateral Sensel Position x")
    ax_profile.set_ylabel("Pressure (kPa / Norm)")
    ax_profile.legend(loc="upper right", framealpha=0.9, fontsize=9)
    ax_profile.grid(True, linestyle=":", alpha=0.6)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, bbox_inches="tight")
    return fig


def plot_confusion_matrix(
    cm: np.ndarray,
    class_names: Optional[List[str]] = None,
    title: str = "Posture Classification Confusion Matrix",
    save_path: Optional[str] = None
) -> plt.Figure:
    """Plots normalized confusion matrix heatmap."""
    if class_names is None:
        class_names = POSTURE_NAMES

    # Normalize by row (true class)
    cm_norm = cm.astype(np.float32) / (cm.sum(axis=1, keepdims=True) + 1e-6)

    fig, ax = plt.subplots(figsize=(7, 6), dpi=150)
    sns.heatmap(
        cm_norm,
        annot=True,
        fmt=".1%",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
        cbar_kws={"label": "Normalized Recall"},
        ax=ax
    )
    ax.set_title(title, fontsize=12, fontweight="bold", pad=12)
    ax.set_xlabel("Predicted Posture", fontsize=11, fontweight="bold")
    ax.set_ylabel("True Posture", fontsize=11, fontweight="bold")
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, bbox_inches="tight")
    return fig


def plot_landmark_error_distribution(
    per_landmark_errors: np.ndarray,
    landmark_names: Optional[List[str]] = None,
    threshold_px: float = 10.0,
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Plots boxplot of Euclidean distance errors per landmark, highlighting PCK threshold.
    
    Args:
        per_landmark_errors: (N, 9) array of Euclidean errors in pixels
    """
    if landmark_names is None:
        landmark_names = LANDMARK_NAMES

    fig, ax = plt.subplots(figsize=(10, 5), dpi=150)
    sns.boxplot(
        data=per_landmark_errors,
        palette="viridis",
        showmeans=True,
        meanprops={"marker": "o", "markerfacecolor": "red", "markeredgecolor": "black", "markersize": 6},
        ax=ax
    )
    ax.axhline(threshold_px, color="red", linestyle="--", linewidth=1.5, label=f"PCK@{threshold_px:.0f}px Threshold")
    ax.axhline(5.0, color="orange", linestyle=":", linewidth=1.5, label="Strict PCK@5px Threshold")

    ax.set_xticklabels(landmark_names, rotation=35, ha="right", fontsize=9)
    ax.set_ylabel("Euclidean Pixel Error (sensels)", fontsize=11, fontweight="bold")
    ax.set_title("Landmark Estimation Error Distribution Across Test Cohort", fontsize=12, fontweight="bold")
    ax.legend(loc="upper right", framealpha=0.9, fontsize=9)
    ax.grid(True, linestyle=":", alpha=0.5, axis="y")
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, bbox_inches="tight")
    return fig


def plot_uncertainty_calibration(
    predicted_uncertainty: np.ndarray,
    observed_errors: np.ndarray,
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Scatter plot with calibration trendline:
    Validates that predicted Bayesian standard deviation sigma correlates strongly
    with actual landmark Euclidean error.
    """
    unc_flat = predicted_uncertainty.flatten()
    err_flat = observed_errors.flatten()

    fig, ax = plt.subplots(figsize=(7, 6), dpi=150)
    # Hexbin / Scatter with alpha
    ax.scatter(unc_flat, err_flat, alpha=0.35, color="#1f77b4", edgecolors="none", s=20, label="Keypoint Estimates")

    # Binned quantile calibration trendline
    bins = np.linspace(np.percentile(unc_flat, 2), np.percentile(unc_flat, 98), 10)
    bin_centers = 0.5 * (bins[:-1] + bins[1:])
    bin_indices = np.digitize(unc_flat, bins)
    binned_err_mean = [
        np.mean(err_flat[bin_indices == i]) if np.sum(bin_indices == i) > 5 else np.nan
        for i in range(1, len(bins))
    ]

    ax.plot(bin_centers, binned_err_mean, "r-o", linewidth=2.5, markersize=8, label="Empirical Calibration Mean")

    # Linear regression fit
    valid = ~np.isnan(unc_flat) & ~np.isnan(err_flat)
    slope, intercept = np.polyfit(unc_flat[valid], err_flat[valid], 1)
    corr = np.corrcoef(unc_flat[valid], err_flat[valid])[0, 1]
    xs = np.linspace(unc_flat.min(), unc_flat.max(), 100)
    ax.plot(xs, slope * xs + intercept, "k--", alpha=0.7, label=f"Fit (r = {corr:.3f})")

    ax.set_title("Bayesian Uncertainty Calibration (MC Dropout)", fontsize=12, fontweight="bold")
    ax.set_xlabel("Predicted Spatial Standard Deviation $\\sigma_k$ (sensels)", fontsize=11, fontweight="bold")
    ax.set_ylabel("True Euclidean Error $\\Vert \\hat{p}_k - p_k \\Vert_2$ (sensels)", fontsize=11, fontweight="bold")
    ax.legend(loc="upper left", framealpha=0.9, fontsize=9)
    ax.grid(True, linestyle=":", alpha=0.5)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, bbox_inches="tight")
    return fig


def plot_loso_summary(
    fold_posture_accuracies: List[float],
    fold_pck_scores: List[float],
    subject_descriptions: Optional[List[str]] = None,
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Plots cross-subject performance across all Leave-One-Subject-Out (LOSO) folds.
    """
    num_folds = len(fold_posture_accuracies)
    if subject_descriptions is None:
        subject_descriptions = [f"Subj {i}" for i in range(num_folds)]

    x = np.arange(num_folds)
    width = 0.38

    fig, ax = plt.subplots(figsize=(11, 5), dpi=150)
    rects1 = ax.bar(x - width / 2, fold_posture_accuracies, width, label="Posture Accuracy (%)", color="#2b5c8f")
    rects2 = ax.bar(x + width / 2, fold_pck_scores, width, label="Landmark PCK@10px (%)", color="#e27c38")

    mean_acc = np.mean(fold_posture_accuracies)
    mean_pck = np.mean(fold_pck_scores)
    ax.axhline(mean_acc, color="#2b5c8f", linestyle="--", linewidth=1.5, label=f"Mean Acc: {mean_acc:.1f}%")
    ax.axhline(mean_pck, color="#e27c38", linestyle="--", linewidth=1.5, label=f"Mean PCK: {mean_pck:.1f}%")

    ax.set_ylabel("Performance (%)", fontsize=11, fontweight="bold")
    ax.set_title("Leave-One-Subject-Out (LOSO) Cross-Subject Generalisation", fontsize=12, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(subject_descriptions, rotation=25, ha="right", fontsize=9)
    ax.set_ylim(60.0, 102.0)
    ax.legend(loc="lower right", framealpha=0.9, fontsize=9)
    ax.grid(True, linestyle=":", alpha=0.5, axis="y")
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, bbox_inches="tight")
    return fig
