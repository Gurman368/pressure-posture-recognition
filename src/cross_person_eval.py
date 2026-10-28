"""
Leave-One-Subject-Out (LOSO) Cross-Validation Pipeline.

Simulates rigorous cross-person generalization validation:
Evaluating whether models trained on N-1 subjects generalize accurately to an unseen,
novel human body with distinct anthropometry, BMI, and load distribution characteristics.
"""

from typing import Dict, List, Optional, Tuple
import os
import json
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset

from src.data_generator import SubjectAnthropometry, get_default_subjects, LANDMARK_NAMES
from src.preprocessing import PressurePreprocessor
from src.models import (
    PostureClassifier,
    LandmarkRegressor,
    TactilePressureDataset,
    compute_pck,
    compute_mean_pixel_error
)
from src.train import (
    train_classifier_epoch,
    evaluate_classifier,
    train_regressor_epoch,
    evaluate_regressor,
    load_config
)
from src.visualise import plot_loso_summary


class LOSOEvaluator:
    """
    Executes Leave-One-Subject-Out Cross-Validation across all subjects.
    """

    def __init__(
        self,
        config: Dict,
        dataset: TactilePressureDataset,
        subjects: List[SubjectAnthropometry],
        device: str = "cpu"
    ):
        self.config = config
        self.dataset = dataset
        self.subjects = subjects
        self.device = torch.device(device)
        self.subject_ids = dataset.subject_ids.numpy()

    def run_fold(
        self,
        held_out_subject_id: int,
        epochs_clf: int = 15,
        epochs_reg: int = 18,
        batch_size: int = 32
    ) -> Dict:
        """
        Runs a single LOSO fold holding out held_out_subject_id.
        """
        train_indices = np.where(self.subject_ids != held_out_subject_id)[0]
        test_indices = np.where(self.subject_ids == held_out_subject_id)[0]

        train_loader = DataLoader(
            Subset(self.dataset, train_indices),
            batch_size=batch_size,
            shuffle=True
        )
        test_loader = DataLoader(
            Subset(self.dataset, test_indices),
            batch_size=batch_size,
            shuffle=False
        )

        subject_info = self.subjects[held_out_subject_id]
        bmi = subject_info.weight_kg / ((subject_info.height_cm / 100.0) ** 2)

        # 1. Train Classifier
        clf = PostureClassifier(
            in_channels=self.config["model"]["classifier"]["in_channels"],
            base_filters=self.config["model"]["classifier"]["base_filters"],
            num_classes=self.config["model"]["classifier"]["num_classes"],
            mc_dropout_prob=self.config["model"]["classifier"]["mc_dropout_prob"]
        ).to(self.device)

        opt_clf = torch.optim.AdamW(clf.parameters(), lr=0.0012, weight_decay=1e-4)
        crit_clf = nn.CrossEntropyLoss()

        for _ in range(epochs_clf):
            train_classifier_epoch(clf, train_loader, opt_clf, crit_clf, self.device)

        clf_metrics = evaluate_classifier(clf, test_loader, self.device)

        # 2. Train Regressor
        reg = LandmarkRegressor(
            in_channels=self.config["model"]["regressor"]["in_channels"],
            base_filters=self.config["model"]["regressor"]["base_filters"],
            posture_cond_dim=self.config["model"]["regressor"]["posture_cond_dim"],
            num_landmarks=self.config["model"]["regressor"]["num_landmarks"],
            grid_height=self.config["dataset"]["grid_height"],
            grid_width=self.config["dataset"]["grid_width"],
            mc_dropout_prob=self.config["model"]["regressor"]["mc_dropout_prob"]
        ).to(self.device)

        opt_reg = torch.optim.AdamW(reg.parameters(), lr=0.001, weight_decay=1e-4)
        crit_reg = nn.SmoothL1Loss()

        for _ in range(epochs_reg):
            train_regressor_epoch(reg, train_loader, opt_reg, crit_reg, self.device)

        reg_metrics = evaluate_regressor(reg, test_loader, self.device, pck_threshold_px=10.0)

        fold_result = {
            "held_out_subject_id": held_out_subject_id,
            "subject_gender": subject_info.gender,
            "subject_height_cm": subject_info.height_cm,
            "subject_weight_kg": subject_info.weight_kg,
            "subject_bmi": float(bmi),
            "posture_accuracy": clf_metrics["accuracy"],
            "posture_f1": clf_metrics["f1_score"],
            "pck@10px": reg_metrics["pck@10px"],
            "pck@5px": reg_metrics["pck@5px"],
            "mean_pixel_error": reg_metrics["mean_pixel_error"],
            "per_landmark_error": reg_metrics["per_landmark_error"].tolist()
        }

        return fold_result

    def run_full_loso(
        self,
        epochs_clf: int = 15,
        epochs_reg: int = 18,
        save_plot_path: Optional[str] = "assets/loso_generalisation_summary.png"
    ) -> Dict:
        """
        Executes all folds and aggregates cross-person generalization performance.
        """
        num_subjects = len(self.subjects)
        print(f"\n=======================================================")
        print(f"Starting Leave-One-Subject-Out (LOSO) Cross-Validation ({num_subjects} Folds)")
        print(f"=======================================================")

        fold_results = []
        for s_idx in range(num_subjects):
            subj = self.subjects[s_idx]
            print(f"\n[Fold {s_idx + 1}/{num_subjects}] Holding out Subject {s_idx} "
                  f"({subj.gender}, {subj.height_cm:.0f}cm, {subj.weight_kg:.0f}kg, BMI: {subj.weight_kg / (subj.height_cm/100)**2:.1f})...")
            
            res = self.run_fold(s_idx, epochs_clf=epochs_clf, epochs_reg=epochs_reg)
            fold_results.append(res)
            print(f"--> Posture Acc: {res['posture_accuracy']:.2f}% | PCK@10px: {res['pck@10px']:.2f}% | Mean Error: {res['mean_pixel_error']:.2f} px")

        # Aggregate metrics
        accuracies = [r["posture_accuracy"] for r in fold_results]
        f1_scores = [r["posture_f1"] for r in fold_results]
        pcks_10 = [r["pck@10px"] for r in fold_results]
        pcks_5 = [r["pck@5px"] for r in fold_results]
        pixel_errors = [r["mean_pixel_error"] for r in fold_results]

        summary = {
            "num_folds": num_subjects,
            "posture_accuracy_mean": float(np.mean(accuracies)),
            "posture_accuracy_std": float(np.std(accuracies)),
            "posture_f1_mean": float(np.mean(f1_scores)),
            "posture_f1_std": float(np.std(f1_scores)),
            "pck@10px_mean": float(np.mean(pcks_10)),
            "pck@10px_std": float(np.std(pcks_10)),
            "pck@5px_mean": float(np.mean(pcks_5)),
            "pck@5px_std": float(np.std(pcks_5)),
            "mean_pixel_error_mean": float(np.mean(pixel_errors)),
            "mean_pixel_error_std": float(np.std(pixel_errors)),
            "fold_details": fold_results
        }

        print("\n=======================================================")
        print("LOSO Cross-Person Validation Summary Across All Subjects:")
        print(f"Posture Accuracy:   {summary['posture_accuracy_mean']:.2f}% ± {summary['posture_accuracy_std']:.2f}%")
        print(f"Posture F1-Score:   {summary['posture_f1_mean']:.2f}% ± {summary['posture_f1_std']:.2f}%")
        print(f"Landmark PCK@10px:  {summary['pck@10px_mean']:.2f}% ± {summary['pck@10px_std']:.2f}%")
        print(f"Landmark PCK@5px:   {summary['pck@5px_mean']:.2f}% ± {summary['pck@5px_std']:.2f}%")
        print(f"Landmark Pixel Err: {summary['mean_pixel_error_mean']:.2f} ± {summary['mean_pixel_error_std']:.2f} px")
        print("=======================================================")

        if save_plot_path:
            os.makedirs(os.path.dirname(save_plot_path), exist_ok=True)
            labels = [
                f"S{r['held_out_subject_id']} ({r['subject_gender'][0].upper()},{r['subject_bmi']:.1f})"
                for r in fold_results
            ]
            plot_loso_summary(accuracies, pcks_10, subject_descriptions=labels, save_path=save_plot_path)
            print(f"Saved LOSO summary chart to: {save_plot_path}")

        return summary
