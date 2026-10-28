"""
Comprehensive End-to-End Research Analysis Script:
Pressure-Based Posture Recognition & Body Landmark Estimation.

Performs:
1. Synthetic Tactile Mat Generation across 10 Anthropometric Subjects & 6 Postures
2. Signal Processing: Inverse Compliant Layer Deconvolution & Dead Sensel Inpainting
3. 94 Biomechanical Load Distribution Feature Extraction
4. Model Training & Validation (PostureClassifier + LandmarkRegressor)
5. Bayesian Uncertainty Quantification (Predictive Entropy & Occlusion Analysis via MC Dropout)
6. Cross-Person Generalisation Evaluation (Leave-One-Subject-Out LOSO)
7. Generation of All Publication Figures in assets/
8. Export of Final Metrics to experiments/results.json
"""

import os
import sys
import json
import time
from typing import Dict, List, Tuple
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset

# Ensure project root is in sys.path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.data_generator import (
    PressureDataGenerator,
    get_default_subjects,
    POSTURE_NAMES,
    LANDMARK_NAMES
)
from src.preprocessing import PressurePreprocessor
from src.models import (
    PostureClassifier,
    LandmarkRegressor,
    TactilePressureDataset,
    compute_pck,
    compute_mean_pixel_error
)
from src.uncertainty import BayesianTactilePredictor, analyze_occlusion_uncertainty
from src.visualise import (
    plot_pressure_with_landmarks,
    plot_deconvolution_comparison,
    plot_confusion_matrix,
    plot_landmark_error_distribution,
    plot_uncertainty_calibration,
    plot_loso_summary
)
from src.train import (
    load_config,
    train_classifier_epoch,
    evaluate_classifier,
    train_regressor_epoch,
    evaluate_regressor
)


def run_full_analysis():
    print("=" * 80)
    print("PRESSURE-POSTURE-RECOGNITION: RESEARCH EXPERIMENT & VALIDATION SUITE")
    print("Intelligent Robotic Surfaces (Beds & Chairs) - Adaptive Tactile Sensing")
    print("=" * 80)

    config_path = os.path.join(PROJECT_ROOT, "experiments", "config.yaml")
    config = load_config(config_path)

    assets_dir = os.path.join(PROJECT_ROOT, "assets")
    checkpoints_dir = os.path.join(PROJECT_ROOT, "experiments", "checkpoints")
    os.makedirs(assets_dir, exist_ok=True)
    os.makedirs(checkpoints_dir, exist_ok=True)

    device = torch.device("cpu")
    print(f"Using Compute Device: {device}")

    # =========================================================================
    # 1. DATA GENERATION
    # =========================================================================
    print("\n" + "-" * 40)
    print("STAGE 1: GENERATING SYNTHETIC TACTILE DATASET")
    print("-" * 40)
    subjects = get_default_subjects()
    print(f"Synthesizing 64x32 sensor arrays across {len(subjects)} anthropometrically distinct subjects:")
    for s in subjects:
        bmi = s.weight_kg / ((s.height_cm / 100.0) ** 2)
        print(f"  - Subject {s.subject_id} ({s.gender:6s}): Height {s.height_cm:.0f}cm, "
              f"Weight {s.weight_kg:.0f}kg (BMI {bmi:4.1f}), Shoulders {s.shoulder_width_cm}cm, Hips {s.hip_width_cm}cm")

    generator = PressureDataGenerator(
        grid_height=config["dataset"]["grid_height"],
        grid_width=config["dataset"]["grid_width"],
        psf_sigma=config["noise_simulation"]["psf_sigma"],
        psf_kernel_size=config["noise_simulation"]["psf_kernel_size"],
        sensor_noise_std=config["noise_simulation"]["sensor_noise_std"],
        quantization_bits=config["noise_simulation"]["quantization_bits"],
        dead_pixel_prob=config["noise_simulation"]["dead_pixel_prob"],
        random_seed=config["dataset"]["random_seed"]
    )

    frames_per_posture = config["dataset"].get("frames_per_posture", 60)
    raw_dataset = generator.generate_dataset(subjects=subjects, frames_per_posture=frames_per_posture)
    num_frames = len(raw_dataset["pressure_maps"])

    # =========================================================================
    # 2. SIGNAL PROCESSING & INVERSE COMPLIANT LAYER DECONVOLUTION
    # =========================================================================
    print("\n" + "-" * 40)
    print("STAGE 2: PREPROCESSING & WIENER DECONVOLUTION")
    print("-" * 40)
    preprocessor = PressurePreprocessor(
        psf_sigma=config["noise_simulation"]["psf_sigma"],
        psf_kernel_size=config["noise_simulation"]["psf_kernel_size"],
        wiener_snr_db=config["preprocessing"]["wiener_snr_db"],
        normalization_method=config["preprocessing"]["normalization"]
    )

    sample_idx = 10  # Supine frame of Subject 0
    raw_sample = raw_dataset["pressure_maps"][sample_idx]
    clean_sample = raw_dataset["clean_maps"][sample_idx]
    dead_mask_sample = raw_dataset["dead_masks"][sample_idx]

    inpainted_sample = preprocessor.inpaint_dead_pixels(raw_sample, dead_mask_sample)
    deconvolved_sample = preprocessor.wiener_deconvolution(inpainted_sample)
    normalized_sample = preprocessor.normalize(deconvolved_sample)

    # Quantify peak sharpening and error reduction
    # Relative MSE to true contact
    mse_raw = np.mean((raw_sample - clean_sample) ** 2)
    mse_deconv = np.mean((deconvolved_sample - clean_sample) ** 2)
    print(f"Mattress Inverse Problem Inversion Results:")
    print(f"  - Raw Degraded vs True Contact MSE:        {mse_raw:.5f}")
    print(f"  - Wiener Deconvolved vs True Contact MSE:  {mse_deconv:.5f}")
    print(f"  - Error Reduction Ratio:                   {(1.0 - mse_deconv / mse_raw) * 100.0:.2f}%")

    # Generate Figure 1: Deconvolution Comparison
    deconv_fig_path = os.path.join(assets_dir, "mattress_deconvolution_comparison.png")
    plot_deconvolution_comparison(
        raw_sample,
        deconvolved_sample,
        clean_sample,
        slice_row=34,
        save_path=deconv_fig_path
    )
    print(f"  - Saved Figure: {deconv_fig_path}")

    # Process all dataset frames
    print("Applying preprocessing pipeline across all dataset frames...")
    processed_maps = np.zeros_like(raw_dataset["pressure_maps"])
    for i in range(num_frames):
        processed_maps[i] = preprocessor.preprocess(
            raw_dataset["pressure_maps"][i],
            dead_mask=raw_dataset["dead_masks"][i]
        )

    # =========================================================================
    # 3. 94 BIOMECHANICAL FEATURES EXTRACTION
    # =========================================================================
    print("\n" + "-" * 40)
    print("STAGE 3: 94 BIOMECHANICAL FEATURE EXTRACTION (IDENTITY & LOAD)")
    print("-" * 40)
    feats = preprocessor.extract_94_biomechanical_features(processed_maps[0])
    print(f"Successfully extracted {len(feats)} biomechanical features per frame.")
    print(f"Feature categories: Regional loads (24), Moments & CoP (9), Gradients (10),")
    print(f"Entropy/Gini (6), Histogram/Sparsity (15), Bilateral Asymmetry (8), 2D Spectral (22).")

    # =========================================================================
    # 4. DATASET PREPARATION & TRAIN/TEST SPLIT
    # =========================================================================
    print("\n" + "-" * 40)
    print("STAGE 4: MODEL TRAINING & VALIDATION")
    print("-" * 40)
    tactile_dataset = TactilePressureDataset(
        pressure_maps=processed_maps,
        landmarks=raw_dataset["landmarks"],
        postures=raw_dataset["postures"],
        subject_ids=raw_dataset["subject_ids"]
    )

    # Partition subjects: 8 subjects (0-7) for training, 2 unseen subjects (8, 9) for test
    sub_ids = tactile_dataset.subject_ids.numpy()
    train_indices = np.where(sub_ids < 8)[0]
    test_indices = np.where(sub_ids >= 8)[0]

    train_loader = DataLoader(Subset(tactile_dataset, train_indices), batch_size=32, shuffle=True)
    test_loader = DataLoader(Subset(tactile_dataset, test_indices), batch_size=32, shuffle=False)
    print(f"Training set: {len(train_indices)} frames (Subjects 0-7)")
    print(f"Test set:     {len(test_indices)} frames (Subjects 8-9, Unseen Novel Bodies)")

    # 4.1 Posture Classifier
    print("\nTraining PostureClassifier (Spatial Attention CNN with MC Dropout)...")
    clf = PostureClassifier(
        in_channels=1,
        base_filters=config["model"]["classifier"]["base_filters"],
        num_classes=6,
        mc_dropout_prob=0.30
    ).to(device)

    opt_clf = torch.optim.AdamW(clf.parameters(), lr=0.0012, weight_decay=1e-4)
    crit_clf = nn.CrossEntropyLoss()

    epochs_clf = 18
    for ep in range(1, epochs_clf + 1):
        loss = train_classifier_epoch(clf, train_loader, opt_clf, crit_clf, device)
        if ep % 6 == 0 or ep == epochs_clf:
            eval_metrics = evaluate_classifier(clf, test_loader, device)
            print(f"  Epoch {ep:02d}/{epochs_clf:02d} | Train Loss: {loss:.4f} | Test Acc: {eval_metrics['accuracy']:.2f}% | Test F1: {eval_metrics['f1_score']:.2f}%")

    test_clf_results = evaluate_classifier(clf, test_loader, device)

    # Figure 2: Confusion Matrix
    cm_path = os.path.join(assets_dir, "posture_confusion_matrix.png")
    plot_confusion_matrix(test_clf_results["confusion_matrix"], POSTURE_NAMES, save_path=cm_path)
    print(f"  - Saved Figure: {cm_path}")

    # 4.2 Landmark Regressor
    print("\nTraining LandmarkRegressor (Posture-Conditioned Deep Architecture)...")
    reg = LandmarkRegressor(
        in_channels=1,
        base_filters=config["model"]["regressor"]["base_filters"],
        posture_cond_dim=config["model"]["regressor"]["posture_cond_dim"],
        num_landmarks=9,
        grid_height=64,
        grid_width=32,
        mc_dropout_prob=0.25
    ).to(device)

    opt_reg = torch.optim.AdamW(reg.parameters(), lr=0.001, weight_decay=1e-4)
    crit_reg = nn.SmoothL1Loss()

    epochs_reg = 22
    for ep in range(1, epochs_reg + 1):
        loss = train_regressor_epoch(reg, train_loader, opt_reg, crit_reg, device)
        if ep % 5 == 0 or ep == epochs_reg:
            eval_reg = evaluate_regressor(reg, test_loader, device, pck_threshold_px=10.0)
            print(f"  Epoch {ep:02d}/{epochs_reg:02d} | Train Loss: {loss:.5f} | Test PCK@10px: {eval_reg['pck@10px']:.2f}% | Pixel Error: {eval_reg['mean_pixel_error']:.2f} px")

    test_reg_results = evaluate_regressor(reg, test_loader, device, pck_threshold_px=10.0)

    # Figure 3: Landmark Error Distribution Boxplot
    dist_fig_path = os.path.join(assets_dir, "landmark_error_distribution.png")
    per_lm_errors_matrix = np.sqrt(
        np.sum((test_reg_results["pred_coords_px"] - test_reg_results["gt_coords_px"]) ** 2, axis=-1)
    )
    plot_landmark_error_distribution(per_lm_errors_matrix, LANDMARK_NAMES, threshold_px=10.0, save_path=dist_fig_path)
    print(f"  - Saved Figure: {dist_fig_path}")

    # =========================================================================
    # 5. BAYESIAN UNCERTAINTY QUANTIFICATION (MC DROPOUT)
    # =========================================================================
    print("\n" + "-" * 40)
    print("STAGE 5: BAYESIAN UNCERTAINTY QUANTIFICATION (MC DROPOUT)")
    print("-" * 40)
    predictor = BayesianTactilePredictor(clf, reg, num_mc_samples=30, device="cpu")

    # Evaluate uncertainty on test batch
    test_pressure_subset = torch.from_numpy(processed_maps[test_indices[:120]]).unsqueeze(1).float()
    test_posture_subset = torch.from_numpy(raw_dataset["postures"][test_indices[:120]]).long()
    test_gt_lm = raw_dataset["landmarks"][test_indices[:120]]

    print("Running 30 Monte Carlo forward passes to quantify predictive entropy and spatial variance...")
    posture_unc = predictor.predict_posture(test_pressure_subset)
    landmark_unc = predictor.predict_landmarks(test_pressure_subset, test_posture_subset)

    # Figure 4: Uncertainty Calibration Plot
    calib_path = os.path.join(assets_dir, "uncertainty_calibration.png")
    pred_spatial_unc = landmark_unc["spatial_uncertainty"]  # (120, 9)
    actual_euclidean_err = np.sqrt(np.sum((landmark_unc["mean_coords_px"] - test_gt_lm) ** 2, axis=-1))
    plot_uncertainty_calibration(pred_spatial_unc, actual_euclidean_err, save_path=calib_path)
    print(f"  - Saved Figure: {calib_path}")

    # Figure 5 & 6: Qualitative Heatmaps with GT/Pred Skeletons & Uncertainty Ellipses
    # Select Supine frame and Left Lateral frame
    supine_local_idx = 0
    lateral_local_idx = 40  # In test set, lateral posture

    supine_fig_path = os.path.join(assets_dir, "tactile_landmarks_supine.png")
    plot_pressure_with_landmarks(
        test_pressure_subset[supine_local_idx, 0].numpy(),
        gt_landmarks=test_gt_lm[supine_local_idx],
        pred_landmarks=landmark_unc["mean_coords_px"][supine_local_idx],
        covariances=landmark_unc["covariances"][supine_local_idx],
        title="Supine Posture: Tactile Heatmap & Landmark Uncertainty",
        save_path=supine_fig_path
    )
    print(f"  - Saved Figure: {supine_fig_path}")

    lateral_fig_path = os.path.join(assets_dir, "tactile_landmarks_lateral.png")
    plot_pressure_with_landmarks(
        test_pressure_subset[lateral_local_idx, 0].numpy(),
        gt_landmarks=test_gt_lm[lateral_local_idx],
        pred_landmarks=landmark_unc["mean_coords_px"][lateral_local_idx],
        covariances=landmark_unc["covariances"][lateral_local_idx],
        title="Left Lateral Posture: Tactile Heatmap & Occluded Limb Uncertainty",
        save_path=lateral_fig_path
    )
    print(f"  - Saved Figure: {lateral_fig_path}")

    # Occlusion Analysis
    occlusion_report = analyze_occlusion_uncertainty(
        pred_spatial_unc,
        test_posture_subset.numpy()
    )
    print("\nOcclusion Uncertainty Analysis (Why robotic surfaces must quantify uncertainty):")
    if "left_lateral" in occlusion_report:
        lat_unc = occlusion_report["left_lateral"]
        print("  In 'Left Lateral' Posture:")
        print(f"    - Contact Side (Left Hip / Trochanter):  Uncertainty = {lat_unc['left_hip']:.2f} px")
        print(f"    - Elevated Side (Right Hip / Occluded):  Uncertainty = {lat_unc['right_hip']:.2f} px")
        print(f"    - Contact Side (Left Knee):              Uncertainty = {lat_unc['left_knee']:.2f} px")
        print(f"    - Elevated Side (Right Knee / Occluded): Uncertainty = {lat_unc['right_knee']:.2f} px")
        ratio = lat_unc['right_hip'] / max(1e-4, lat_unc['left_hip'])
        print(f"    --> Occluded landmarks show {ratio:.2f}x higher spatial uncertainty!")

    # =========================================================================
    # 6. LEAVE-ONE-SUBJECT-OUT (LOSO) CROSS-PERSON EVALUATION
    # =========================================================================
    print("\n" + "-" * 40)
    print("STAGE 6: LEAVE-ONE-SUBJECT-OUT (LOSO) VALIDATION")
    print("-" * 40)
    from src.cross_person_eval import LOSOEvaluator
    loso_evaluator = LOSOEvaluator(config, tactile_dataset, subjects, device="cpu")
    loso_summary = loso_evaluator.run_full_loso(
        epochs_clf=12,
        epochs_reg=15,
        save_plot_path=os.path.join(assets_dir, "loso_generalisation_summary.png")
    )

    # =========================================================================
    # 7. EXPORT METRICS & FINAL REPORT
    # =========================================================================
    results_export = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "dataset_summary": {
            "num_subjects": len(subjects),
            "num_postures": 6,
            "total_frames": num_frames,
            "grid_dimensions": [64, 32]
        },
        "deconvolution_metrics": {
            "mse_raw": float(mse_raw),
            "mse_deconvolved": float(mse_deconv),
            "improvement_pct": float((1.0 - mse_deconv / mse_raw) * 100.0)
        },
        "validation_metrics_unseen_subjects": {
            "posture_accuracy": test_clf_results["accuracy"],
            "posture_f1_score": test_clf_results["f1_score"],
            "pck@10px": test_reg_results["pck@10px"],
            "pck@5px": test_reg_results["pck@5px"],
            "mean_pixel_error": test_reg_results["mean_pixel_error"]
        },
        "loso_cross_validation": {
            "posture_accuracy_mean": loso_summary["posture_accuracy_mean"],
            "posture_accuracy_std": loso_summary["posture_accuracy_std"],
            "pck@10px_mean": loso_summary["pck@10px_mean"],
            "pck@10px_std": loso_summary["pck@10px_std"],
            "mean_pixel_error_mean": loso_summary["mean_pixel_error_mean"],
            "mean_pixel_error_std": loso_summary["mean_pixel_error_std"]
        }
    }

    results_file_path = os.path.join(PROJECT_ROOT, "experiments", "results.json")
    with open(results_file_path, "w") as f:
        json.dump(results_export, f, indent=2)
    print(f"\nFinal experiment results exported to: {results_file_path}")
    print("\nALL RESEARCH EVALUATION STEPS COMPLETED SUCCESSFULLY!")


if __name__ == "__main__":
    run_full_analysis()
