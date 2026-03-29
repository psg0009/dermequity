"""
DermEquity: Fairness Auditing Toolkit for Dermatological AI
============================================================

A comprehensive framework for auditing and mitigating skin tone bias 
in dermatological AI systems.

Author: Parth Gosar
Affiliation: Thomas Lord Department of Computer Science, USC
License: MIT

Features:
---------
- Comprehensive fairness audit by Fitzpatrick skin type (I-VI)
- Five mitigation strategies including Synthetic LAB Augmentation
- Counterfactual analysis for causal bias detection
- GradCAM explainability visualization
- MC Dropout uncertainty quantification
- Selective prediction with deferral

Usage:
------
    from dermequity import (
        FairnessAuditor, 
        MitigationStrategies,
        CounterfactualAnalyzer,
        GradCAMExplainer,
        UncertaintyEstimator,
        SelectivePredictor
    )
    
    # Initialize auditor
    auditor = FairnessAuditor(model, device='cuda')
    results = auditor.audit(test_loader)
    
    # Counterfactual analysis
    analyzer = CounterfactualAnalyzer(model, device='cuda')
    flip_rate = analyzer.analyze(test_loader)
    
    # GradCAM visualization
    explainer = GradCAMExplainer(model, target_layer='layer4')
    heatmap = explainer.generate(image)
"""

import os
import cv2
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Optional, Union, Callable
from dataclasses import dataclass, field
from collections import defaultdict
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, WeightedRandomSampler, Dataset
import torchvision.transforms as transforms

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, roc_curve
)


# =============================================================================
# DATA CLASSES
# =============================================================================

@dataclass
class FairnessMetrics:
    """Container for fairness metrics."""
    accuracy: float
    sensitivity: float  # True Positive Rate (Recall)
    specificity: float  # True Negative Rate
    ppv: float  # Positive Predictive Value (Precision)
    npv: float  # Negative Predictive Value
    f1: float
    auc: Optional[float] = None
    n_samples: int = 0


@dataclass
class EquityGap:
    """Container for equity gap results."""
    metric: str
    gap: float
    best_group: int
    worst_group: int
    best_value: float
    worst_value: float
    
    def __str__(self):
        return (f"{self.metric.upper()}: Gap={self.gap:.1%} "
                f"(Best: Type {self.best_group}={self.best_value:.3f}, "
                f"Worst: Type {self.worst_group}={self.worst_value:.3f})")


@dataclass
class AuditResults:
    """Complete audit results."""
    metrics_by_fitzpatrick: Dict[int, FairnessMetrics]
    overall_metrics: FairnessMetrics
    equity_gaps: Dict[str, EquityGap]
    predictions_df: pd.DataFrame


@dataclass
class CounterfactualResults:
    """Results from counterfactual analysis."""
    flip_rate: float
    flip_rate_by_fst: Dict[int, float]
    avg_prob_change: float
    avg_prob_change_by_fst: Dict[int, float]
    details_df: pd.DataFrame


@dataclass
class UncertaintyResults:
    """Results from uncertainty analysis."""
    avg_confidence: float
    avg_entropy: float
    confidence_by_fst: Dict[int, float]
    entropy_by_fst: Dict[int, float]
    calibration_error: float
    details_df: pd.DataFrame


# =============================================================================
# CORE METRICS FUNCTIONS
# =============================================================================

class BiasMetrics:
    """Calculate fairness metrics for binary classification."""
    
    @staticmethod
    def compute_specificity(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Compute specificity (True Negative Rate)."""
        tn = np.sum((y_true == 0) & (y_pred == 0))
        fp = np.sum((y_true == 0) & (y_pred == 1))
        return tn / (tn + fp) if (tn + fp) > 0 else 0.0
    
    @staticmethod
    def compute_npv(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Compute Negative Predictive Value."""
        tn = np.sum((y_true == 0) & (y_pred == 0))
        fn = np.sum((y_true == 1) & (y_pred == 0))
        return tn / (tn + fn) if (tn + fn) > 0 else 0.0
    
    @staticmethod
    def compute_all_metrics(
        y_true: np.ndarray, 
        y_pred: np.ndarray,
        y_prob: Optional[np.ndarray] = None
    ) -> FairnessMetrics:
        """Compute all fairness metrics."""
        y_true = np.array(y_true)
        y_pred = np.array(y_pred)
        
        # Handle edge cases
        if len(y_true) == 0:
            return FairnessMetrics(0, 0, 0, 0, 0, 0, None, 0)
        
        accuracy = accuracy_score(y_true, y_pred)
        
        # Sensitivity (Recall / TPR)
        sensitivity = recall_score(y_true, y_pred, zero_division=0)
        
        # Specificity (TNR)
        specificity = BiasMetrics.compute_specificity(y_true, y_pred)
        
        # PPV (Precision)
        ppv = precision_score(y_true, y_pred, zero_division=0)
        
        # NPV
        npv = BiasMetrics.compute_npv(y_true, y_pred)
        
        # F1
        f1 = f1_score(y_true, y_pred, zero_division=0)
        
        # AUC (if probabilities provided)
        auc = None
        if y_prob is not None and len(np.unique(y_true)) > 1:
            try:
                auc = roc_auc_score(y_true, y_prob)
            except ValueError:
                auc = None
        
        return FairnessMetrics(
            accuracy=accuracy,
            sensitivity=sensitivity,
            specificity=specificity,
            ppv=ppv,
            npv=npv,
            f1=f1,
            auc=auc,
            n_samples=len(y_true)
        )


# =============================================================================
# FAIRNESS AUDITOR
# =============================================================================

class FairnessAuditor:
    """Main class for auditing model fairness across Fitzpatrick skin types."""
    
    def __init__(self, model: nn.Module, device: str = 'cuda'):
        """
        Initialize the fairness auditor.
        
        Parameters
        ----------
        model : nn.Module
            PyTorch model to audit
        device : str
            Device to run inference on ('cuda' or 'cpu')
        """
        self.model = model
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        self.model.eval()
        self.results = None
    
    def audit(
        self,
        test_loader: DataLoader,
        fitzpatrick_idx: int = 2
    ) -> AuditResults:
        """
        Run fairness audit on test data.
        
        Parameters
        ----------
        test_loader : DataLoader
            Test data loader. Each batch should yield (images, labels, fitzpatrick_types)
        fitzpatrick_idx : int
            Index of fitzpatrick label in batch tuple (default: 2)
            
        Returns
        -------
        AuditResults
            Complete audit results with metrics by skin type
        """
        all_preds, all_labels, all_fitz, all_probs = [], [], [], []
        
        with torch.no_grad():
            for batch in test_loader:
                images = batch[0].to(self.device)
                labels = batch[1]
                fitz_types = batch[fitzpatrick_idx]
                
                outputs = self.model(images)
                probs = torch.softmax(outputs, dim=1)[:, 1].cpu().numpy()
                preds = (probs >= 0.5).astype(int)
                
                all_preds.extend(preds)
                all_probs.extend(probs)
                all_labels.extend(labels.numpy())
                all_fitz.extend(fitz_types if isinstance(fitz_types, list) else fitz_types.numpy())
        
        # Create DataFrame
        predictions_df = pd.DataFrame({
            'prediction': all_preds,
            'probability': all_probs,
            'true_label': all_labels,
            'fitzpatrick': all_fitz
        })
        
        # Calculate metrics by Fitzpatrick type
        metrics_by_fst = {}
        for fst in sorted(predictions_df['fitzpatrick'].unique()):
            subset = predictions_df[predictions_df['fitzpatrick'] == fst]
            metrics_by_fst[fst] = BiasMetrics.compute_all_metrics(
                subset['true_label'].values,
                subset['prediction'].values,
                subset['probability'].values
            )
        
        # Overall metrics
        overall_metrics = BiasMetrics.compute_all_metrics(
            predictions_df['true_label'].values,
            predictions_df['prediction'].values,
            predictions_df['probability'].values
        )
        
        # Calculate equity gaps
        equity_gaps = self._calculate_equity_gaps(metrics_by_fst)
        
        self.results = AuditResults(
            metrics_by_fitzpatrick=metrics_by_fst,
            overall_metrics=overall_metrics,
            equity_gaps=equity_gaps,
            predictions_df=predictions_df
        )
        
        return self.results
    
    def _calculate_equity_gaps(
        self, 
        metrics_by_fst: Dict[int, FairnessMetrics]
    ) -> Dict[str, EquityGap]:
        """Calculate equity gaps for all metrics."""
        gaps = {}
        metric_names = ['accuracy', 'sensitivity', 'specificity', 'ppv', 'f1']
        
        for metric in metric_names:
            values = {fst: getattr(m, metric) for fst, m in metrics_by_fst.items()}
            best_fst = max(values, key=values.get)
            worst_fst = min(values, key=values.get)
            
            gaps[metric] = EquityGap(
                metric=metric,
                gap=values[best_fst] - values[worst_fst],
                best_group=best_fst,
                worst_group=worst_fst,
                best_value=values[best_fst],
                worst_value=values[worst_fst]
            )
        
        # AUC gap (if available)
        auc_values = {fst: m.auc for fst, m in metrics_by_fst.items() if m.auc is not None}
        if auc_values:
            best_fst = max(auc_values, key=auc_values.get)
            worst_fst = min(auc_values, key=auc_values.get)
            gaps['auc'] = EquityGap(
                metric='auc',
                gap=auc_values[best_fst] - auc_values[worst_fst],
                best_group=best_fst,
                worst_group=worst_fst,
                best_value=auc_values[best_fst],
                worst_value=auc_values[worst_fst]
            )
        
        return gaps
    
    def print_report(self):
        """Print formatted audit report."""
        if self.results is None:
            print("No audit results. Run audit() first.")
            return
        
        print("=" * 80)
        print("DERMEQUITY FAIRNESS AUDIT REPORT")
        print("=" * 80)
        
        # Overall performance
        print("\n📊 OVERALL PERFORMANCE:")
        print("-" * 40)
        om = self.results.overall_metrics
        print(f"   Accuracy:    {om.accuracy:.3f}")
        print(f"   Sensitivity: {om.sensitivity:.3f}")
        print(f"   Specificity: {om.specificity:.3f}")
        print(f"   PPV:         {om.ppv:.3f}")
        print(f"   F1:          {om.f1:.3f}")
        if om.auc:
            print(f"   AUC:         {om.auc:.3f}")
        
        # By Fitzpatrick type
        print("\n📊 PERFORMANCE BY FITZPATRICK SKIN TYPE:")
        print("-" * 70)
        print(f"{'Type':<8}{'N':<8}{'Acc':<10}{'Sens':<10}{'Spec':<10}{'PPV':<10}{'AUC':<10}")
        print("-" * 70)
        
        for fst, m in sorted(self.results.metrics_by_fitzpatrick.items()):
            auc_str = f"{m.auc:.3f}" if m.auc else "N/A"
            print(f"{fst:<8}{m.n_samples:<8}{m.accuracy:<10.3f}{m.sensitivity:<10.3f}"
                  f"{m.specificity:<10.3f}{m.ppv:<10.3f}{auc_str:<10}")
        
        # Equity gaps
        print("\n🚨 EQUITY GAPS:")
        print("-" * 70)
        for name, gap in self.results.equity_gaps.items():
            severity = "🔴 HIGH" if gap.gap > 0.2 else "🟡 MEDIUM" if gap.gap > 0.1 else "🟢 LOW"
            print(f"   {severity} {gap}")
        
        print("=" * 80)
    
    def plot_results(self, save_path: Optional[str] = None, figsize: Tuple[int, int] = (15, 10)):
        """Generate visualization of audit results."""
        if self.results is None:
            print("No audit results. Run audit() first.")
            return
        
        fig, axes = plt.subplots(2, 3, figsize=figsize)
        fig.suptitle('DermEquity Fairness Audit Results', fontsize=14, fontweight='bold')
        
        fitz_types = sorted(self.results.metrics_by_fitzpatrick.keys())
        metrics = ['accuracy', 'sensitivity', 'specificity', 'ppv', 'f1', 'auc']
        titles = ['Accuracy', 'Sensitivity (TPR)', 'Specificity (TNR)', 
                  'PPV (Precision)', 'F1 Score', 'AUC-ROC']
        
        for idx, (metric, title) in enumerate(zip(metrics, titles)):
            ax = axes[idx // 3, idx % 3]
            values = [getattr(self.results.metrics_by_fitzpatrick[f], metric) or 0 
                      for f in fitz_types]
            
            colors = plt.cm.RdYlGn([v for v in values])
            bars = ax.bar(range(len(fitz_types)), values, color=colors)
            
            ax.set_title(title, fontweight='bold')
            ax.set_xlabel('Fitzpatrick Skin Type')
            ax.set_ylabel(title)
            ax.set_xticks(range(len(fitz_types)))
            ax.set_xticklabels([f'Type {f}' for f in fitz_types])
            ax.set_ylim(0, 1.1)
            ax.axhline(y=np.mean(values), color='red', linestyle='--', 
                       label=f'Mean: {np.mean(values):.3f}')
            ax.legend(loc='lower right')
            
            # Add value labels
            for bar, val in zip(bars, values):
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                        f'{val:.2f}', ha='center', va='bottom', fontsize=9)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Saved plot to {save_path}")
        
        plt.show()
        return fig
    
    def to_dict(self) -> dict:
        """Export results to dictionary."""
        if self.results is None:
            return {}
        
        return {
            'overall': {
                'accuracy': self.results.overall_metrics.accuracy,
                'sensitivity': self.results.overall_metrics.sensitivity,
                'specificity': self.results.overall_metrics.specificity,
                'ppv': self.results.overall_metrics.ppv,
                'f1': self.results.overall_metrics.f1,
                'auc': self.results.overall_metrics.auc,
                'n_samples': self.results.overall_metrics.n_samples
            },
            'by_fitzpatrick': {
                fst: {
                    'accuracy': m.accuracy,
                    'sensitivity': m.sensitivity,
                    'specificity': m.specificity,
                    'ppv': m.ppv,
                    'f1': m.f1,
                    'auc': m.auc,
                    'n_samples': m.n_samples
                }
                for fst, m in self.results.metrics_by_fitzpatrick.items()
            },
            'equity_gaps': {
                name: {
                    'gap': gap.gap,
                    'best_group': gap.best_group,
                    'worst_group': gap.worst_group,
                    'best_value': gap.best_value,
                    'worst_value': gap.worst_value
                }
                for name, gap in self.results.equity_gaps.items()
            }
        }
    
    def save_results(self, path: str):
        """Save results to JSON file."""
        with open(path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)
        print(f"Results saved to {path}")


# =============================================================================
# SYNTHETIC LAB AUGMENTATION
# =============================================================================

class SyntheticLABAugmentation:
    """
    Synthetic skin tone augmentation using LAB color space.
    
    This is the best-performing mitigation strategy (68% gap reduction).
    Modifies the L-channel (luminance) to simulate different skin tones
    while preserving lesion morphology.
    """
    
    def __init__(self, l_shift_range: Tuple[int, int] = (-30, 30)):
        """
        Initialize LAB augmentation.
        
        Parameters
        ----------
        l_shift_range : Tuple[int, int]
            Range for L-channel shift (default: -30 to +30)
        """
        self.l_shift_range = l_shift_range
    
    def augment(
        self, 
        image: np.ndarray, 
        l_shift: Optional[int] = None
    ) -> np.ndarray:
        """
        Apply LAB augmentation to an image.
        
        Parameters
        ----------
        image : np.ndarray
            Input image in RGB format (H, W, 3), uint8
        l_shift : int, optional
            Specific L-channel shift. If None, randomly sampled from range.
            
        Returns
        -------
        np.ndarray
            Augmented image in RGB format
        """
        if l_shift is None:
            l_shift = np.random.randint(self.l_shift_range[0], self.l_shift_range[1] + 1)
        
        # Convert RGB to LAB
        lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
        
        # Modify L channel
        l_channel = lab[:, :, 0].astype(np.int16)
        l_channel = np.clip(l_channel + l_shift, 0, 255).astype(np.uint8)
        lab[:, :, 0] = l_channel
        
        # Convert back to RGB
        augmented = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
        
        return augmented
    
    def generate_variants(
        self, 
        image: np.ndarray, 
        n_variants: int = 5
    ) -> List[np.ndarray]:
        """
        Generate multiple skin tone variants of an image.
        
        Parameters
        ----------
        image : np.ndarray
            Input image in RGB format
        n_variants : int
            Number of variants to generate
            
        Returns
        -------
        List[np.ndarray]
            List of augmented images
        """
        shifts = np.linspace(self.l_shift_range[0], self.l_shift_range[1], n_variants)
        return [self.augment(image, int(shift)) for shift in shifts]
    
    def __call__(self, image: np.ndarray) -> np.ndarray:
        """Apply random LAB augmentation."""
        return self.augment(image)


class SyntheticLABTransform:
    """PyTorch-compatible transform for LAB augmentation."""
    
    def __init__(self, l_shift_range: Tuple[int, int] = (-30, 30), p: float = 0.5):
        """
        Parameters
        ----------
        l_shift_range : Tuple[int, int]
            Range for L-channel shift
        p : float
            Probability of applying augmentation
        """
        self.augmenter = SyntheticLABAugmentation(l_shift_range)
        self.p = p
    
    def __call__(self, image):
        """Apply transform to PIL Image or tensor."""
        if np.random.random() > self.p:
            return image
        
        # Convert to numpy if needed
        if isinstance(image, Image.Image):
            img_array = np.array(image)
            augmented = self.augmenter.augment(img_array)
            return Image.fromarray(augmented)
        elif isinstance(image, torch.Tensor):
            img_array = image.permute(1, 2, 0).numpy()
            img_array = (img_array * 255).astype(np.uint8)
            augmented = self.augmenter.augment(img_array)
            return torch.from_numpy(augmented).permute(2, 0, 1).float() / 255
        else:
            return self.augmenter.augment(image)


# =============================================================================
# COUNTERFACTUAL ANALYSIS
# =============================================================================

class CounterfactualAnalyzer:
    """
    Counterfactual analysis for causal bias detection.
    
    Modifies skin tone in images and measures prediction changes
    to establish causal relationship between skin tone and model output.
    """
    
    def __init__(
        self, 
        model: nn.Module, 
        device: str = 'cuda',
        l_shifts: List[int] = [-30, -20, -10, 10, 20, 30]
    ):
        """
        Initialize counterfactual analyzer.
        
        Parameters
        ----------
        model : nn.Module
            PyTorch model to analyze
        device : str
            Device for inference
        l_shifts : List[int]
            L-channel shifts to test
        """
        self.model = model
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        self.model.eval()
        self.l_shifts = l_shifts
        self.augmenter = SyntheticLABAugmentation()
        self.results = None
    
    def analyze(
        self,
        test_loader: DataLoader,
        transform: Optional[Callable] = None,
        fitzpatrick_idx: int = 2
    ) -> CounterfactualResults:
        """
        Run counterfactual analysis on test data.
        
        Parameters
        ----------
        test_loader : DataLoader
            Test data loader yielding (images, labels, fitzpatrick_types)
        transform : Callable, optional
            Transform to apply to augmented images before inference
        fitzpatrick_idx : int
            Index of fitzpatrick label in batch
            
        Returns
        -------
        CounterfactualResults
            Analysis results including flip rates
        """
        results = []
        
        # Default transform
        if transform is None:
            transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])
        
        with torch.no_grad():
            for batch in test_loader:
                images = batch[0]  # Keep on CPU for augmentation
                labels = batch[1]
                fitz_types = batch[fitzpatrick_idx]
                
                for i in range(len(images)):
                    # Get original prediction
                    img_tensor = images[i:i+1].to(self.device)
                    orig_output = self.model(img_tensor)
                    orig_prob = torch.softmax(orig_output, dim=1)[0, 1].item()
                    orig_pred = int(orig_prob >= 0.5)
                    
                    # Convert image to numpy for augmentation
                    img_np = images[i].permute(1, 2, 0).numpy()
                    img_np = (img_np * 255).astype(np.uint8)
                    
                    # Test counterfactuals
                    cf_probs = []
                    cf_preds = []
                    
                    for l_shift in self.l_shifts:
                        # Augment image
                        aug_img = self.augmenter.augment(img_np, l_shift)
                        
                        # Transform and predict
                        aug_pil = Image.fromarray(aug_img)
                        aug_tensor = transform(aug_pil).unsqueeze(0).to(self.device)
                        
                        cf_output = self.model(aug_tensor)
                        cf_prob = torch.softmax(cf_output, dim=1)[0, 1].item()
                        cf_pred = int(cf_prob >= 0.5)
                        
                        cf_probs.append(cf_prob)
                        cf_preds.append(cf_pred)
                    
                    # Calculate metrics
                    prediction_flipped = any(p != orig_pred for p in cf_preds)
                    prob_range = max(cf_probs) - min(cf_probs)
                    
                    results.append({
                        'original_prob': orig_prob,
                        'original_pred': orig_pred,
                        'true_label': labels[i].item(),
                        'fitzpatrick': fitz_types[i] if isinstance(fitz_types[i], int) else fitz_types[i].item(),
                        'prediction_flipped': prediction_flipped,
                        'prob_range': prob_range,
                        'min_prob': min(cf_probs),
                        'max_prob': max(cf_probs),
                        'cf_probs': cf_probs
                    })
        
        # Create DataFrame
        details_df = pd.DataFrame(results)
        
        # Calculate overall metrics
        flip_rate = details_df['prediction_flipped'].mean()
        avg_prob_change = details_df['prob_range'].mean()
        
        # By Fitzpatrick type
        flip_rate_by_fst = details_df.groupby('fitzpatrick')['prediction_flipped'].mean().to_dict()
        avg_prob_change_by_fst = details_df.groupby('fitzpatrick')['prob_range'].mean().to_dict()
        
        self.results = CounterfactualResults(
            flip_rate=flip_rate,
            flip_rate_by_fst=flip_rate_by_fst,
            avg_prob_change=avg_prob_change,
            avg_prob_change_by_fst=avg_prob_change_by_fst,
            details_df=details_df
        )
        
        return self.results
    
    def visualize_counterfactual(
        self,
        image: np.ndarray,
        save_path: Optional[str] = None
    ) -> plt.Figure:
        """
        Visualize counterfactual variants of a single image.
        
        Parameters
        ----------
        image : np.ndarray
            Input image in RGB format
        save_path : str, optional
            Path to save figure
            
        Returns
        -------
        plt.Figure
            Matplotlib figure
        """
        shifts = [-30, -20, -10, 0, 10, 20, 30]
        n_cols = len(shifts)
        
        fig, axes = plt.subplots(1, n_cols, figsize=(3 * n_cols, 3))
        fig.suptitle('Counterfactual Skin Tone Variants', fontweight='bold')
        
        for idx, shift in enumerate(shifts):
            if shift == 0:
                variant = image
                title = 'Original'
            else:
                variant = self.augmenter.augment(image, shift)
                title = f'L-shift: {shift:+d}'
            
            axes[idx].imshow(variant)
            axes[idx].set_title(title)
            axes[idx].axis('off')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def print_report(self):
        """Print counterfactual analysis report."""
        if self.results is None:
            print("No results. Run analyze() first.")
            return
        
        print("=" * 60)
        print("COUNTERFACTUAL ANALYSIS REPORT")
        print("=" * 60)
        
        print(f"\n📊 OVERALL RESULTS:")
        print(f"   Prediction Flip Rate: {self.results.flip_rate:.1%}")
        print(f"   Avg Probability Range: {self.results.avg_prob_change:.3f}")
        
        print(f"\n📊 BY FITZPATRICK TYPE:")
        print("-" * 40)
        print(f"{'Type':<10}{'Flip Rate':<15}{'Prob Range':<15}")
        print("-" * 40)
        
        for fst in sorted(self.results.flip_rate_by_fst.keys()):
            flip = self.results.flip_rate_by_fst.get(fst, 0)
            prob = self.results.avg_prob_change_by_fst.get(fst, 0)
            print(f"{fst:<10}{flip:<15.1%}{prob:<15.3f}")
        
        print("=" * 60)


# =============================================================================
# GRADCAM EXPLAINABILITY
# =============================================================================

class GradCAMExplainer:
    """
    GradCAM visualization for model explainability.
    
    Generates attention heatmaps to understand where the model
    focuses when making predictions.
    """
    
    def __init__(
        self, 
        model: nn.Module, 
        target_layer: str = 'layer4',
        device: str = 'cuda'
    ):
        """
        Initialize GradCAM explainer.
        
        Parameters
        ----------
        model : nn.Module
            PyTorch model (should be a CNN like ResNet, EfficientNet)
        target_layer : str
            Name of the layer to compute GradCAM on
        device : str
            Device for inference
        """
        self.model = model
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        self.model.eval()
        self.target_layer = target_layer
        
        # Hooks for gradients and activations
        self.gradients = None
        self.activations = None
        self._register_hooks()
    
    def _register_hooks(self):
        """Register forward and backward hooks on target layer."""
        def forward_hook(module, input, output):
            self.activations = output.detach()
        
        def backward_hook(module, grad_in, grad_out):
            self.gradients = grad_out[0].detach()
        
        # Find target layer
        for name, module in self.model.named_modules():
            if name == self.target_layer:
                module.register_forward_hook(forward_hook)
                module.register_full_backward_hook(backward_hook)
                return
        
        raise ValueError(f"Layer '{self.target_layer}' not found in model")
    
    def generate(
        self, 
        image: torch.Tensor, 
        target_class: Optional[int] = None
    ) -> np.ndarray:
        """
        Generate GradCAM heatmap for an image.
        
        Parameters
        ----------
        image : torch.Tensor
            Input image tensor (1, C, H, W) or (C, H, W)
        target_class : int, optional
            Class to generate heatmap for. If None, uses predicted class.
            
        Returns
        -------
        np.ndarray
            GradCAM heatmap (H, W) normalized to [0, 1]
        """
        if image.dim() == 3:
            image = image.unsqueeze(0)
        
        image = image.to(self.device)
        image.requires_grad = True
        
        # Forward pass
        output = self.model(image)
        
        if target_class is None:
            target_class = output.argmax(dim=1).item()
        
        # Backward pass
        self.model.zero_grad()
        one_hot = torch.zeros_like(output)
        one_hot[0, target_class] = 1
        output.backward(gradient=one_hot)
        
        # Compute GradCAM
        weights = self.gradients.mean(dim=(2, 3), keepdim=True)
        cam = (weights * self.activations).sum(dim=1, keepdim=True)
        cam = F.relu(cam)
        
        # Normalize and resize
        cam = cam.squeeze().cpu().numpy()
        cam = cv2.resize(cam, (image.shape[3], image.shape[2]))
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        
        return cam
    
    def visualize(
        self,
        image: Union[torch.Tensor, np.ndarray],
        heatmap: Optional[np.ndarray] = None,
        alpha: float = 0.4,
        save_path: Optional[str] = None
    ) -> plt.Figure:
        """
        Visualize GradCAM heatmap overlaid on image.
        
        Parameters
        ----------
        image : torch.Tensor or np.ndarray
            Original image
        heatmap : np.ndarray, optional
            Pre-computed heatmap. If None, generates new one.
        alpha : float
            Transparency of heatmap overlay
        save_path : str, optional
            Path to save figure
            
        Returns
        -------
        plt.Figure
            Matplotlib figure
        """
        # Convert image to numpy if needed
        if isinstance(image, torch.Tensor):
            if image.dim() == 4:
                image = image.squeeze(0)
            img_np = image.permute(1, 2, 0).cpu().numpy()
            # Denormalize
            mean = np.array([0.485, 0.456, 0.406])
            std = np.array([0.229, 0.224, 0.225])
            img_np = img_np * std + mean
            img_np = np.clip(img_np, 0, 1)
        else:
            img_np = image / 255.0 if image.max() > 1 else image
        
        # Generate heatmap if not provided
        if heatmap is None:
            if isinstance(image, np.ndarray):
                # Convert numpy to tensor
                transform = transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                ])
                image_tensor = transform(Image.fromarray((image * 255).astype(np.uint8)))
            else:
                image_tensor = image
            heatmap = self.generate(image_tensor)
        
        # Create visualization
        fig, axes = plt.subplots(1, 3, figsize=(12, 4))
        
        # Original image
        axes[0].imshow(img_np)
        axes[0].set_title('Original Image')
        axes[0].axis('off')
        
        # Heatmap
        axes[1].imshow(heatmap, cmap='jet')
        axes[1].set_title('GradCAM Heatmap')
        axes[1].axis('off')
        
        # Overlay
        heatmap_colored = plt.cm.jet(heatmap)[:, :, :3]
        overlay = img_np * (1 - alpha) + heatmap_colored * alpha
        axes[2].imshow(overlay)
        axes[2].set_title('Overlay')
        axes[2].axis('off')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def analyze_by_fitzpatrick(
        self,
        test_loader: DataLoader,
        fitzpatrick_idx: int = 2,
        n_samples_per_type: int = 10
    ) -> Dict[int, Dict[str, float]]:
        """
        Analyze GradCAM patterns across Fitzpatrick types.
        
        Parameters
        ----------
        test_loader : DataLoader
            Test data loader
        fitzpatrick_idx : int
            Index of fitzpatrick label in batch
        n_samples_per_type : int
            Number of samples to analyze per type
            
        Returns
        -------
        Dict[int, Dict[str, float]]
            Metrics by Fitzpatrick type (entropy, concentration, confidence)
        """
        results = defaultdict(list)
        samples_per_type = defaultdict(int)
        
        for batch in test_loader:
            images = batch[0].to(self.device)
            fitz_types = batch[fitzpatrick_idx]
            
            for i in range(len(images)):
                fst = fitz_types[i] if isinstance(fitz_types[i], int) else fitz_types[i].item()
                
                if samples_per_type[fst] >= n_samples_per_type:
                    continue
                
                # Generate heatmap
                heatmap = self.generate(images[i:i+1])
                
                # Compute metrics
                # Entropy (higher = more diffuse attention)
                heatmap_norm = heatmap / (heatmap.sum() + 1e-8)
                entropy = -np.sum(heatmap_norm * np.log(heatmap_norm + 1e-8))
                
                # Concentration (% of attention in top 10% of pixels)
                threshold = np.percentile(heatmap, 90)
                concentration = heatmap[heatmap >= threshold].sum() / (heatmap.sum() + 1e-8)
                
                # Peak value
                peak = heatmap.max()
                
                results[fst].append({
                    'entropy': entropy,
                    'concentration': concentration,
                    'peak': peak
                })
                samples_per_type[fst] += 1
            
            # Check if we have enough samples
            if all(samples_per_type[f] >= n_samples_per_type 
                   for f in range(1, 7) if f in samples_per_type):
                break
        
        # Aggregate
        aggregated = {}
        for fst, metrics_list in results.items():
            aggregated[fst] = {
                'entropy': np.mean([m['entropy'] for m in metrics_list]),
                'concentration': np.mean([m['concentration'] for m in metrics_list]),
                'peak': np.mean([m['peak'] for m in metrics_list])
            }
        
        return aggregated


# =============================================================================
# MC DROPOUT UNCERTAINTY
# =============================================================================

class UncertaintyEstimator:
    """
    Monte Carlo Dropout for uncertainty quantification.
    
    Performs multiple forward passes with dropout enabled
    to estimate prediction uncertainty.
    """
    
    def __init__(
        self, 
        model: nn.Module, 
        n_forward_passes: int = 20,
        device: str = 'cuda'
    ):
        """
        Initialize uncertainty estimator.
        
        Parameters
        ----------
        model : nn.Module
            PyTorch model (should have dropout layers)
        n_forward_passes : int
            Number of stochastic forward passes
        device : str
            Device for inference
        """
        self.model = model
        self.n_forward_passes = n_forward_passes
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        self.results = None
    
    def _enable_dropout(self):
        """Enable dropout layers during inference."""
        for module in self.model.modules():
            if isinstance(module, nn.Dropout):
                module.train()
    
    def estimate(
        self,
        test_loader: DataLoader,
        fitzpatrick_idx: int = 2
    ) -> UncertaintyResults:
        """
        Estimate uncertainty on test data.
        
        Parameters
        ----------
        test_loader : DataLoader
            Test data loader
        fitzpatrick_idx : int
            Index of fitzpatrick label in batch
            
        Returns
        -------
        UncertaintyResults
            Uncertainty analysis results
        """
        self.model.eval()
        self._enable_dropout()
        
        results = []
        
        with torch.no_grad():
            for batch in test_loader:
                images = batch[0].to(self.device)
                labels = batch[1]
                fitz_types = batch[fitzpatrick_idx]
                
                for i in range(len(images)):
                    # Multiple forward passes
                    probs = []
                    for _ in range(self.n_forward_passes):
                        output = self.model(images[i:i+1])
                        prob = torch.softmax(output, dim=1)[0, 1].item()
                        probs.append(prob)
                    
                    probs = np.array(probs)
                    
                    # Mean prediction
                    mean_prob = probs.mean()
                    pred = int(mean_prob >= 0.5)
                    
                    # Uncertainty metrics
                    std = probs.std()
                    confidence = max(mean_prob, 1 - mean_prob)
                    
                    # Predictive entropy
                    entropy = -(mean_prob * np.log(mean_prob + 1e-8) + 
                               (1 - mean_prob) * np.log(1 - mean_prob + 1e-8))
                    
                    results.append({
                        'mean_prob': mean_prob,
                        'std': std,
                        'confidence': confidence,
                        'entropy': entropy,
                        'prediction': pred,
                        'true_label': labels[i].item(),
                        'correct': pred == labels[i].item(),
                        'fitzpatrick': fitz_types[i] if isinstance(fitz_types[i], int) else fitz_types[i].item()
                    })
        
        # Create DataFrame
        details_df = pd.DataFrame(results)
        
        # Calculate calibration error
        calibration_error = self._calculate_ece(details_df)
        
        # Aggregate metrics
        self.results = UncertaintyResults(
            avg_confidence=details_df['confidence'].mean(),
            avg_entropy=details_df['entropy'].mean(),
            confidence_by_fst=details_df.groupby('fitzpatrick')['confidence'].mean().to_dict(),
            entropy_by_fst=details_df.groupby('fitzpatrick')['entropy'].mean().to_dict(),
            calibration_error=calibration_error,
            details_df=details_df
        )
        
        return self.results
    
    def _calculate_ece(self, df: pd.DataFrame, n_bins: int = 10) -> float:
        """Calculate Expected Calibration Error."""
        bins = np.linspace(0, 1, n_bins + 1)
        ece = 0.0
        
        for i in range(n_bins):
            mask = (df['confidence'] >= bins[i]) & (df['confidence'] < bins[i + 1])
            if mask.sum() == 0:
                continue
            
            bin_conf = df.loc[mask, 'confidence'].mean()
            bin_acc = df.loc[mask, 'correct'].mean()
            bin_size = mask.sum() / len(df)
            
            ece += bin_size * abs(bin_acc - bin_conf)
        
        return ece
    
    def print_report(self):
        """Print uncertainty analysis report."""
        if self.results is None:
            print("No results. Run estimate() first.")
            return
        
        print("=" * 60)
        print("UNCERTAINTY ANALYSIS REPORT (MC Dropout)")
        print("=" * 60)
        
        print(f"\n📊 OVERALL METRICS:")
        print(f"   Average Confidence: {self.results.avg_confidence:.1%}")
        print(f"   Average Entropy: {self.results.avg_entropy:.3f}")
        print(f"   Calibration Error (ECE): {self.results.calibration_error:.3f}")
        
        print(f"\n📊 BY FITZPATRICK TYPE:")
        print("-" * 50)
        print(f"{'Type':<10}{'Confidence':<15}{'Entropy':<15}{'Accuracy':<15}")
        print("-" * 50)
        
        acc_by_fst = self.results.details_df.groupby('fitzpatrick')['correct'].mean()
        
        for fst in sorted(self.results.confidence_by_fst.keys()):
            conf = self.results.confidence_by_fst.get(fst, 0)
            ent = self.results.entropy_by_fst.get(fst, 0)
            acc = acc_by_fst.get(fst, 0)
            print(f"{fst:<10}{conf:<15.1%}{ent:<15.3f}{acc:<15.1%}")
        
        print("=" * 60)


# =============================================================================
# SELECTIVE PREDICTION / DEFERRAL
# =============================================================================

class SelectivePredictor:
    """
    Selective prediction with deferral to human review.
    
    Defers low-confidence predictions to improve accuracy
    at the cost of coverage.
    """
    
    def __init__(
        self, 
        model: nn.Module, 
        uncertainty_estimator: Optional[UncertaintyEstimator] = None,
        device: str = 'cuda'
    ):
        """
        Initialize selective predictor.
        
        Parameters
        ----------
        model : nn.Module
            PyTorch model
        uncertainty_estimator : UncertaintyEstimator, optional
            Estimator for MC Dropout uncertainty. If None, uses softmax confidence.
        device : str
            Device for inference
        """
        self.model = model
        self.uncertainty_estimator = uncertainty_estimator
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        self.model.eval()
    
    def evaluate_thresholds(
        self,
        test_loader: DataLoader,
        thresholds: List[float] = [0.5, 0.6, 0.7, 0.8, 0.9, 0.95],
        fitzpatrick_idx: int = 2
    ) -> pd.DataFrame:
        """
        Evaluate performance at different confidence thresholds.
        
        Parameters
        ----------
        test_loader : DataLoader
            Test data loader
        thresholds : List[float]
            Confidence thresholds to evaluate
        fitzpatrick_idx : int
            Index of fitzpatrick label in batch
            
        Returns
        -------
        pd.DataFrame
            Performance metrics at each threshold
        """
        # Get predictions and confidences
        all_preds, all_labels, all_confs, all_fitz = [], [], [], []
        
        with torch.no_grad():
            for batch in test_loader:
                images = batch[0].to(self.device)
                labels = batch[1]
                fitz_types = batch[fitzpatrick_idx]
                
                outputs = self.model(images)
                probs = torch.softmax(outputs, dim=1)
                
                # Confidence = max probability
                confs, preds = probs.max(dim=1)
                
                all_preds.extend(preds.cpu().numpy())
                all_confs.extend(confs.cpu().numpy())
                all_labels.extend(labels.numpy())
                all_fitz.extend(fitz_types if isinstance(fitz_types, list) else fitz_types.numpy())
        
        all_preds = np.array(all_preds)
        all_labels = np.array(all_labels)
        all_confs = np.array(all_confs)
        all_fitz = np.array(all_fitz)
        
        # Evaluate at each threshold
        results = []
        
        for threshold in thresholds:
            mask = all_confs >= threshold
            coverage = mask.mean()
            
            if mask.sum() == 0:
                continue
            
            retained_acc = accuracy_score(all_labels[mask], all_preds[mask])
            retained_sens = recall_score(all_labels[mask], all_preds[mask], zero_division=0)
            
            # Calculate equity gap on retained samples
            gap = 0
            accs_by_fst = {}
            for fst in np.unique(all_fitz):
                fst_mask = mask & (all_fitz == fst)
                if fst_mask.sum() > 0:
                    accs_by_fst[fst] = accuracy_score(all_labels[fst_mask], all_preds[fst_mask])
            
            if accs_by_fst:
                gap = max(accs_by_fst.values()) - min(accs_by_fst.values())
            
            results.append({
                'threshold': threshold,
                'coverage': coverage,
                'accuracy': retained_acc,
                'sensitivity': retained_sens,
                'equity_gap': gap,
                'n_retained': mask.sum(),
                'n_deferred': (~mask).sum()
            })
        
        return pd.DataFrame(results)
    
    def predict_with_deferral(
        self,
        images: torch.Tensor,
        threshold: float = 0.9
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Make predictions with deferral for low-confidence cases.
        
        Parameters
        ----------
        images : torch.Tensor
            Batch of images
        threshold : float
            Confidence threshold for deferral
            
        Returns
        -------
        Tuple[np.ndarray, np.ndarray, np.ndarray]
            (predictions, confidences, should_defer mask)
        """
        self.model.eval()
        images = images.to(self.device)
        
        with torch.no_grad():
            outputs = self.model(images)
            probs = torch.softmax(outputs, dim=1)
            confs, preds = probs.max(dim=1)
        
        preds = preds.cpu().numpy()
        confs = confs.cpu().numpy()
        should_defer = confs < threshold
        
        return preds, confs, should_defer
    
    def plot_coverage_accuracy_curve(
        self,
        test_loader: DataLoader,
        fitzpatrick_idx: int = 2,
        save_path: Optional[str] = None
    ) -> plt.Figure:
        """
        Plot coverage vs accuracy curve.
        
        Parameters
        ----------
        test_loader : DataLoader
            Test data loader
        fitzpatrick_idx : int
            Index of fitzpatrick label in batch
        save_path : str, optional
            Path to save figure
            
        Returns
        -------
        plt.Figure
            Matplotlib figure
        """
        thresholds = np.arange(0.5, 1.0, 0.02)
        df = self.evaluate_thresholds(test_loader, thresholds.tolist(), fitzpatrick_idx)
        
        fig, axes = plt.subplots(1, 3, figsize=(15, 4))
        
        # Coverage vs Accuracy
        axes[0].plot(df['coverage'], df['accuracy'], 'b-o', linewidth=2)
        axes[0].set_xlabel('Coverage')
        axes[0].set_ylabel('Accuracy')
        axes[0].set_title('Coverage vs Accuracy Trade-off')
        axes[0].grid(True, alpha=0.3)
        
        # Threshold vs Coverage
        axes[1].plot(df['threshold'], df['coverage'], 'g-o', linewidth=2)
        axes[1].set_xlabel('Confidence Threshold')
        axes[1].set_ylabel('Coverage')
        axes[1].set_title('Threshold vs Coverage')
        axes[1].grid(True, alpha=0.3)
        
        # Coverage vs Equity Gap
        axes[2].plot(df['coverage'], df['equity_gap'], 'r-o', linewidth=2)
        axes[2].set_xlabel('Coverage')
        axes[2].set_ylabel('Equity Gap')
        axes[2].set_title('Coverage vs Equity Gap')
        axes[2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig


# =============================================================================
# MITIGATION STRATEGIES
# =============================================================================

class MitigationStrategies:
    """Collection of bias mitigation strategies."""
    
    @staticmethod
    def create_balanced_sampler(
        df: pd.DataFrame,
        label_col: str = 'label',
        fitzpatrick_col: str = 'fitzpatrick'
    ) -> WeightedRandomSampler:
        """
        Create a balanced sampler that equalizes representation.
        
        Strategy 1: Balanced Resampling (↓19% gap reduction)
        
        Parameters
        ----------
        df : pd.DataFrame
            Training data with labels and Fitzpatrick types
        label_col : str
            Column name for class labels
        fitzpatrick_col : str
            Column name for Fitzpatrick skin types
            
        Returns
        -------
        WeightedRandomSampler
            Sampler for DataLoader
        """
        # Create combined group (label + fitzpatrick)
        df['_fairness_group'] = (df[label_col].astype(str) + '_fitz' + 
                                  df[fitzpatrick_col].astype(str))
        
        # Calculate weights
        group_counts = df['_fairness_group'].value_counts()
        weights = 1.0 / df['_fairness_group'].map(group_counts)
        
        # Normalize
        weights = weights / weights.sum() * len(weights)
        
        return WeightedRandomSampler(
            weights=weights.values,
            num_samples=len(weights),
            replacement=True
        )
    
    @staticmethod
    def get_skin_tone_aware_transform(
        fitzpatrick_type: int,
        is_train: bool = True,
        base_size: int = 224
    ) -> transforms.Compose:
        """
        Get transforms with intensity based on skin type.
        
        Strategy 2: Skin-Aware Augmentation (↓41% gap reduction)
        
        Parameters
        ----------
        fitzpatrick_type : int
            Fitzpatrick skin type (1-6)
        is_train : bool
            Whether this is for training (apply augmentation)
        base_size : int
            Target image size
            
        Returns
        -------
        transforms.Compose
            PyTorch transforms
        """
        # Higher augmentation intensity for darker skin types
        intensity = 1.0 + (fitzpatrick_type - 1) * 0.2  # 1.0 to 2.0
        
        if is_train:
            return transforms.Compose([
                transforms.RandomResizedCrop(base_size, scale=(0.8 / intensity, 1.0)),
                transforms.RandomHorizontalFlip(),
                transforms.RandomVerticalFlip(),
                transforms.RandomRotation(30 * intensity),
                transforms.ColorJitter(
                    brightness=0.2 * intensity,
                    contrast=0.2 * intensity,
                    saturation=0.2 * intensity,
                    hue=0.1 * intensity
                ),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])
        else:
            return transforms.Compose([
                transforms.Resize(base_size + 32),
                transforms.CenterCrop(base_size),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])
    
    @staticmethod
    def get_synthetic_lab_transform(
        l_shift_range: Tuple[int, int] = (-30, 30),
        p: float = 0.5
    ) -> SyntheticLABTransform:
        """
        Get synthetic LAB augmentation transform.
        
        Strategy 4: Synthetic Skin Tone Augmentation (↓68% gap reduction)
        BEST PERFORMING STRATEGY
        
        Parameters
        ----------
        l_shift_range : Tuple[int, int]
            Range for L-channel shift
        p : float
            Probability of applying augmentation
            
        Returns
        -------
        SyntheticLABTransform
            Transform for LAB augmentation
        """
        return SyntheticLABTransform(l_shift_range, p)
    
    @staticmethod
    def train_ensemble_specialists(
        train_df: pd.DataFrame,
        model_factory: Callable,
        train_fn: Callable,
        fitzpatrick_col: str = 'fitzpatrick',
        groups: Dict[str, List[int]] = None
    ) -> Dict[str, nn.Module]:
        """
        Train ensemble of specialist models.
        
        Strategy 3: Ensemble of Specialists (↓56% gap reduction)
        
        Parameters
        ----------
        train_df : pd.DataFrame
            Training data
        model_factory : callable
            Function that returns a new model instance
        train_fn : callable
            Function that trains a model: train_fn(model, subset_df)
        fitzpatrick_col : str
            Column name for Fitzpatrick types
        groups : Dict[str, List[int]]
            Skin type groupings. Default: light (1-2), medium (3-4), dark (5-6)
            
        Returns
        -------
        Dict[str, nn.Module]
            Dictionary of trained specialist models
        """
        if groups is None:
            groups = {
                'light': [1, 2],
                'medium': [3, 4],
                'dark': [5, 6]
            }
        
        specialists = {}
        
        for group_name, fitz_types in groups.items():
            print(f"Training specialist for {group_name} skin (FST {fitz_types})...")
            
            # Filter data
            subset = train_df[train_df[fitzpatrick_col].isin(fitz_types)]
            
            # Create and train model
            model = model_factory()
            trained_model = train_fn(model, subset)
            
            specialists[group_name] = trained_model
        
        return specialists
    
    @staticmethod
    def ensemble_predict(
        specialists: Dict[str, nn.Module],
        images: torch.Tensor,
        fitzpatrick_types: List[int],
        device: str = 'cuda'
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Make predictions using ensemble of specialists.
        
        Parameters
        ----------
        specialists : Dict[str, nn.Module]
            Dictionary of specialist models
        images : torch.Tensor
            Batch of images
        fitzpatrick_types : List[int]
            Fitzpatrick type for each image
        device : str
            Device for inference
            
        Returns
        -------
        Tuple[np.ndarray, np.ndarray]
            (predictions, probabilities)
        """
        device = torch.device(device if torch.cuda.is_available() else 'cpu')
        
        # Map Fitzpatrick types to groups
        fst_to_group = {
            1: 'light', 2: 'light',
            3: 'medium', 4: 'medium',
            5: 'dark', 6: 'dark'
        }
        
        preds = []
        probs = []
        
        for i in range(len(images)):
            group = fst_to_group.get(fitzpatrick_types[i], 'medium')
            model = specialists[group]
            model.to(device)
            model.eval()
            
            with torch.no_grad():
                output = model(images[i:i+1].to(device))
                prob = torch.softmax(output, dim=1)[0, 1].item()
            
            preds.append(int(prob >= 0.5))
            probs.append(prob)
        
        return np.array(preds), np.array(probs)


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

def compare_models(
    models: Dict[str, nn.Module],
    test_loader: DataLoader,
    device: str = 'cuda'
) -> pd.DataFrame:
    """
    Compare multiple models on fairness metrics.
    
    Parameters
    ----------
    models : Dict[str, nn.Module]
        Dictionary of named models
    test_loader : DataLoader
        Test data loader
    device : str
        Device for inference
        
    Returns
    -------
    pd.DataFrame
        Comparison table
    """
    results = []
    
    for name, model in models.items():
        auditor = FairnessAuditor(model, device=device)
        audit = auditor.audit(test_loader)
        
        results.append({
            'model': name,
            'accuracy': audit.overall_metrics.accuracy,
            'sensitivity': audit.overall_metrics.sensitivity,
            'specificity': audit.overall_metrics.specificity,
            'ppv': audit.overall_metrics.ppv,
            'auc': audit.overall_metrics.auc,
            'accuracy_gap': audit.equity_gaps['accuracy'].gap,
            'sensitivity_gap': audit.equity_gaps['sensitivity'].gap
        })
    
    return pd.DataFrame(results)


def full_analysis(
    model: nn.Module,
    test_loader: DataLoader,
    device: str = 'cuda',
    output_dir: str = './dermequity_results'
) -> Dict:
    """
    Run complete DermEquity analysis pipeline.
    
    Parameters
    ----------
    model : nn.Module
        PyTorch model to analyze
    test_loader : DataLoader
        Test data loader
    device : str
        Device for inference
    output_dir : str
        Directory to save results
        
    Returns
    -------
    Dict
        All analysis results
    """
    os.makedirs(output_dir, exist_ok=True)
    
    results = {}
    
    # 1. Fairness Audit
    print("Running fairness audit...")
    auditor = FairnessAuditor(model, device=device)
    results['audit'] = auditor.audit(test_loader)
    auditor.print_report()
    auditor.plot_results(save_path=os.path.join(output_dir, 'fairness_audit.png'))
    auditor.save_results(os.path.join(output_dir, 'audit_results.json'))
    
    # 2. Counterfactual Analysis
    print("\nRunning counterfactual analysis...")
    cf_analyzer = CounterfactualAnalyzer(model, device=device)
    results['counterfactual'] = cf_analyzer.analyze(test_loader)
    cf_analyzer.print_report()
    
    # 3. Uncertainty Analysis
    print("\nRunning uncertainty analysis...")
    uncertainty = UncertaintyEstimator(model, device=device)
    results['uncertainty'] = uncertainty.estimate(test_loader)
    uncertainty.print_report()
    
    # 4. Selective Prediction
    print("\nEvaluating selective prediction...")
    selector = SelectivePredictor(model, device=device)
    results['selective'] = selector.evaluate_thresholds(test_loader)
    selector.plot_coverage_accuracy_curve(
        test_loader, 
        save_path=os.path.join(output_dir, 'selective_prediction.png')
    )
    
    print(f"\n✅ Analysis complete! Results saved to {output_dir}")
    
    return results


# =============================================================================
# VERSION INFO
# =============================================================================

__version__ = '2.0.0'
__author__ = 'Parth Gosar'
__affiliation__ = 'Thomas Lord Department of Computer Science, USC'
