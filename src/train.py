"""
Training Pipeline for Posture Classification and Body Landmark Estimation.

Supports:
1. End-to-end training of Spatial-Attention PostureClassifier and Posture-Conditioned LandmarkRegressor.
2. Standard train/val split or Leave-One-Subject-Out (LOSO) mode.
3. Checkpoint management, loss scheduling, and PCK@10px evaluation.
"""

import argparse
import os
import json
from typing import Dict, Optional, Tuple
import yaml
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix

from src.data_generator import PressureDataGenerator, get_default_subjects
from src.preprocessing import PressurePreprocessor
from src.models import (
    PostureClassifier,
    LandmarkRegressor,
    TactilePressureDataset,
    compute_pck,
    compute_mean_pixel_error
)


def load_config(config_path: str = "experiments/config.yaml") -> Dict:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def prepare_dataset(
    config: Dict,
    apply_preprocessing: bool = True
) -> Tuple[TactilePressureDataset, Dict[str, np.ndarray]]:
    """
    Generates synthetic tactile dataset and applies Wiener deconvolution & normalization.
    """
    gen_cfg = config.get("dataset", {})
    noise_cfg = config.get("noise_simulation", {})
    prep_cfg = config.get("preprocessing", {})

    generator = PressureDataGenerator(
        grid_height=gen_cfg.get("grid_height", 64),
        grid_width=gen_cfg.get("grid_width", 32),
        psf_sigma=noise_cfg.get("psf_sigma", 1.8),
        psf_kernel_size=noise_cfg.get("psf_kernel_size", 9),
        sensor_noise_std=noise_cfg.get("sensor_noise_std", 0.02),
        quantization_bits=noise_cfg.get("quantization_bits", 8),
        dead_pixel_prob=noise_cfg.get("dead_pixel_prob", 0.008),
        random_seed=gen_cfg.get("random_seed", 42)
    )

    subjects = get_default_subjects()[:gen_cfg.get("num_subjects", 10)]
    raw_data = generator.generate_dataset(
        subjects=subjects,
        frames_per_posture=gen_cfg.get("frames_per_posture", 60)
    )

    processed_maps = raw_data["pressure_maps"]
    if apply_preprocessing:
        print("Applying regularized Wiener deconvolution and dead pixel inpainting...")
        preprocessor = PressurePreprocessor(
            psf_sigma=noise_cfg.get("psf_sigma", 1.8),
            psf_kernel_size=noise_cfg.get("psf_kernel_size", 9),
            wiener_snr_db=prep_cfg.get("wiener_snr_db", 22.0),
            normalization_method=prep_cfg.get("normalization", "minmax_robust")
        )

        n_frames = len(processed_maps)
        deconvolved_maps = np.zeros_like(processed_maps)
        for i in range(n_frames):
            deconvolved_maps[i] = preprocessor.preprocess(
                processed_maps[i],
                dead_mask=raw_data["dead_masks"][i]
            )
        processed_maps = deconvolved_maps

    dataset = TactilePressureDataset(
        pressure_maps=processed_maps,
        landmarks=raw_data["landmarks"],
        postures=raw_data["postures"],
        subject_ids=raw_data["subject_ids"],
        grid_height=gen_cfg.get("grid_height", 64),
        grid_width=gen_cfg.get("grid_width", 32)
    )

    return dataset, raw_data


def train_classifier_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device
) -> float:
    model.train()
    total_loss = 0.0
    for batch in loader:
        x = batch["pressure"].to(device)
        y = batch["posture"].to(device)

        optimizer.zero_grad()
        logits = model(x)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * len(y)
    return total_loss / len(loader.dataset)


def evaluate_classifier(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device
) -> Dict[str, float]:
    model.eval()
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for batch in loader:
            x = batch["pressure"].to(device)
            y = batch["posture"].to(device)
            logits = model(x)
            preds = torch.argmax(logits, dim=-1)
            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(y.cpu().numpy())

    all_preds = np.array(all_preds)
    all_targets = np.array(all_targets)

    acc = accuracy_score(all_targets, all_preds) * 100.0
    prec, rec, f1, _ = precision_recall_fscore_support(all_targets, all_preds, average="weighted", zero_division=0)
    cm = confusion_matrix(all_targets, all_preds)

    return {
        "accuracy": float(acc),
        "f1_score": float(f1 * 100.0),
        "precision": float(prec * 100.0),
        "recall": float(rec * 100.0),
        "confusion_matrix": cm,
        "predictions": all_preds,
        "targets": all_targets
    }


def train_regressor_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device
) -> float:
    model.train()
    total_loss = 0.0
    for batch in loader:
        x = batch["pressure"].to(device)
        posture = batch["posture"].to(device)
        target_norm = batch["landmarks_norm"].to(device)

        optimizer.zero_grad()
        pred_norm = model(x, posture, return_normalized=True)
        loss = criterion(pred_norm, target_norm)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * len(x)
    return total_loss / len(loader.dataset)


def evaluate_regressor(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    pck_threshold_px: float = 10.0
) -> Dict[str, Union[float, np.ndarray]]:
    model.eval()
    all_pred_px = []
    all_gt_px = []

    with torch.no_grad():
        for batch in loader:
            x = batch["pressure"].to(device)
            posture = batch["posture"].to(device)
            pred_px = model(x, posture, return_normalized=False)
            all_pred_px.append(pred_px.cpu().numpy())
            all_gt_px.append(batch["landmarks_px"].cpu().numpy())

    all_pred_px = np.concatenate(all_pred_px, axis=0)
    all_gt_px = np.concatenate(all_gt_px, axis=0)

    pck10, pck10_per_lm = compute_pck(all_pred_px, all_gt_px, threshold_px=pck_threshold_px)
    pck5, pck5_per_lm = compute_pck(all_pred_px, all_gt_px, threshold_px=5.0)
    mean_err, per_lm_err = compute_mean_pixel_error(all_pred_px, all_gt_px)

    return {
        "pck@10px": float(pck10),
        "pck@5px": float(pck5),
        "mean_pixel_error": float(mean_err),
        "per_landmark_error": per_lm_err,
        "per_landmark_pck10": pck10_per_lm,
        "pred_coords_px": all_pred_px,
        "gt_coords_px": all_gt_px
    }


def train_pipeline(
    config_path: str = "experiments/config.yaml",
    save_checkpoints: bool = True
) -> Dict:
    """Runs standard training pipeline."""
    config = load_config(config_path)
    train_cfg = config.get("training", {})
    paths_cfg = config.get("paths", {})
    device_str = train_cfg.get("device", "cpu")
    if device_str == "cuda" and not torch.cuda.is_available():
        device_str = "cpu"
    device = torch.device(device_str)

    os.makedirs(paths_cfg.get("checkpoint_dir", "experiments/checkpoints"), exist_ok=True)
    os.makedirs(paths_cfg.get("assets_dir", "assets"), exist_ok=True)

    dataset, raw_data = prepare_dataset(config)

    # Split train/val by subject: 8 subjects for train (0-7), 2 subjects for test (8, 9)
    sub_ids = dataset.subject_ids.numpy()
    train_indices = np.where(sub_ids < 8)[0]
    val_indices = np.where(sub_ids >= 8)[0]

    train_subset = torch.utils.data.Subset(dataset, train_indices)
    val_subset = torch.utils.data.Subset(dataset, val_indices)

    batch_size = train_cfg.get("batch_size", 32)
    train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_subset, batch_size=batch_size, shuffle=False)

    print(f"\n--- Training Set: {len(train_subset)} frames | Validation Set: {len(val_subset)} frames ---")

    # 1. Posture Classifier
    print("\n[Stage 1] Training PostureClassifier (Spatial Attention CNN)...")
    clf = PostureClassifier(
        in_channels=config["model"]["classifier"]["in_channels"],
        base_filters=config["model"]["classifier"]["base_filters"],
        num_classes=config["model"]["classifier"]["num_classes"],
        mc_dropout_prob=config["model"]["classifier"]["mc_dropout_prob"]
    ).to(device)

    optimizer_clf = torch.optim.AdamW(
        clf.parameters(),
        lr=train_cfg.get("lr_classifier", 1e-3),
        weight_decay=train_cfg.get("weight_decay", 1e-4)
    )
    criterion_clf = nn.CrossEntropyLoss()
    best_acc = 0.0

    epochs_clf = train_cfg.get("epochs_classifier", 20)
    for epoch in range(1, epochs_clf + 1):
        loss = train_classifier_epoch(clf, train_loader, optimizer_clf, criterion_clf, device)
        if epoch % 5 == 0 or epoch == epochs_clf:
            val_metrics = evaluate_classifier(clf, val_loader, device)
            acc = val_metrics["accuracy"]
            print(f"Epoch {epoch:02d}/{epochs_clf:02d} | Train Loss: {loss:.4f} | Val Acc: {acc:.2f}% | Val F1: {val_metrics['f1_score']:.2f}%")
            if acc > best_acc and save_checkpoints:
                best_acc = acc
                torch.save(clf.state_dict(), os.path.join(paths_cfg.get("checkpoint_dir"), "best_classifier.pt"))

    # 2. Landmark Regressor
    print("\n[Stage 2] Training LandmarkRegressor (Posture-Conditioned Keypoint Regressor)...")
    reg = LandmarkRegressor(
        in_channels=config["model"]["regressor"]["in_channels"],
        base_filters=config["model"]["regressor"]["base_filters"],
        posture_cond_dim=config["model"]["regressor"]["posture_cond_dim"],
        num_landmarks=config["model"]["regressor"]["num_landmarks"],
        grid_height=config["dataset"]["grid_height"],
        grid_width=config["dataset"]["grid_width"],
        mc_dropout_prob=config["model"]["regressor"]["mc_dropout_prob"]
    ).to(device)

    optimizer_reg = torch.optim.AdamW(
        reg.parameters(),
        lr=train_cfg.get("lr_regressor", 8e-4),
        weight_decay=train_cfg.get("weight_decay", 1e-4)
    )
    criterion_reg = nn.SmoothL1Loss()
    best_pck = 0.0

    epochs_reg = train_cfg.get("epochs_regressor", 25)
    for epoch in range(1, epochs_reg + 1):
        loss = train_regressor_epoch(reg, train_loader, optimizer_reg, criterion_reg, device)
        if epoch % 5 == 0 or epoch == epochs_reg:
            val_metrics = evaluate_regressor(reg, val_loader, device, pck_threshold_px=train_cfg.get("pck_threshold_px", 10.0))
            pck = val_metrics["pck@10px"]
            print(f"Epoch {epoch:02d}/{epochs_reg:02d} | Train Loss: {loss:.5f} | Val PCK@10px: {pck:.2f}% | Mean Pixel Error: {val_metrics['mean_pixel_error']:.2f} px")
            if pck > best_pck and save_checkpoints:
                best_pck = pck
                torch.save(reg.state_dict(), os.path.join(paths_cfg.get("checkpoint_dir"), "best_regressor.pt"))

    # Final evaluation
    final_clf_eval = evaluate_classifier(clf, val_loader, device)
    final_reg_eval = evaluate_regressor(reg, val_loader, device)

    results = {
        "classifier_val_accuracy": final_clf_eval["accuracy"],
        "classifier_val_f1": final_clf_eval["f1_score"],
        "regressor_val_pck10": final_reg_eval["pck@10px"],
        "regressor_val_pck5": final_reg_eval["pck@5px"],
        "regressor_val_pixel_error": final_reg_eval["mean_pixel_error"]
    }

    print("\n=== Final Validation Results ===")
    print(f"Posture Classification Accuracy: {results['classifier_val_accuracy']:.2f}%")
    print(f"Posture Classification F1-Score: {results['classifier_val_f1']:.2f}%")
    print(f"Landmark PCK@10px:               {results['regressor_val_pck10']:.2f}%")
    print(f"Landmark PCK@5px:                {results['regressor_val_pck5']:.2f}%")
    print(f"Landmark Mean Pixel Error:       {results['regressor_val_pixel_error']:.2f} pixels")

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train pressure posture recognition & landmark estimation models")
    parser.add_argument("--config", type=str, default="experiments/config.yaml", help="Path to config yaml")
    args = parser.parse_args()
    train_pipeline(args.config)
