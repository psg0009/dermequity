# DermEquity API Reference

## Table of Contents

1. [FairnessAuditor](#fairnessauditor)
2. [SyntheticLABAugmentation](#syntheticlabaugmentation)
3. [CounterfactualAnalyzer](#counterfactualanalyzer)
4. [GradCAMExplainer](#gradcamexplainer)
5. [UncertaintyEstimator](#uncertaintyestimator)
6. [SelectivePredictor](#selectivepredictor)
7. [MitigationStrategies](#mitigationstrategies)
8. [Data Classes](#data-classes)

---

## FairnessAuditor

Main class for conducting fairness audits on dermatological AI models.

### Constructor

```python
FairnessAuditor(model: nn.Module, device: str = 'cuda')
```

**Parameters:**
- `model`: PyTorch model to audit
- `device`: Device for inference ('cuda' or 'cpu')

### Methods

#### `audit(test_loader, fitzpatrick_idx=2)`

Run fairness audit on test data.

**Parameters:**
- `test_loader`: DataLoader yielding (images, labels, fitzpatrick_types)
- `fitzpatrick_idx`: Index of Fitzpatrick labels in batch tuple

**Returns:** `AuditResults`

#### `print_report()`

Print formatted audit report to console.

#### `plot_results(save_path=None, figsize=(14, 10))`

Generate visualization of audit results.

**Parameters:**
- `save_path`: Optional path to save figure
- `figsize`: Figure size tuple

**Returns:** matplotlib Figure

#### `to_dict()`

Export results as dictionary.

**Returns:** dict

#### `save_results(path)`

Save results to JSON file.

---

## SyntheticLABAugmentation

Synthetic skin tone augmentation using LAB color space. **Key mitigation strategy achieving 68% gap reduction.**

### Constructor

```python
SyntheticLABAugmentation(l_shift_range: Tuple[int, int] = (-30, 30))
```

**Parameters:**
- `l_shift_range`: Range for L-channel shift (default: -30 to +30)

### Methods

#### `augment(image, l_shift=None)`

Apply skin tone augmentation to a single image.

**Parameters:**
- `image`: Input image (H, W, 3), RGB, 0-255
- `l_shift`: Specific L-channel shift. If None, randomly sampled.

**Returns:** Augmented image (H, W, 3)

#### `augment_batch(images, l_shifts=None)`

Apply augmentation to a batch of images.

**Parameters:**
- `images`: Batch tensor (B, C, H, W), normalized [0, 1]
- `l_shifts`: Optional list of shifts per image

**Returns:** Augmented batch tensor

#### `create_variants(image, n_variants=5)`

Create multiple skin tone variants of an image.

**Parameters:**
- `image`: Input image (H, W, 3)
- `n_variants`: Number of variants to generate

**Returns:** List of (augmented_image, l_shift) tuples

---

## CounterfactualAnalyzer

Counterfactual analysis for causal bias detection.

### Constructor

```python
CounterfactualAnalyzer(
    model: nn.Module, 
    device: str = 'cuda',
    l_shifts: List[int] = [-30, -20, -10, 10, 20, 30]
)
```

**Parameters:**
- `model`: Model to analyze
- `device`: Device for inference
- `l_shifts`: L-channel shifts to test

### Methods

#### `analyze(test_loader, fitzpatrick_idx=2)`

Run counterfactual analysis.

**Returns:** `CounterfactualResults`

#### `print_report()`

Print analysis report.

#### `visualize_example(image, save_path=None)`

Visualize counterfactual variants for a single image.

**Returns:** matplotlib Figure

---

## GradCAMExplainer

Grad-CAM visualization for model explainability.

### Constructor

```python
GradCAMExplainer(
    model: nn.Module, 
    target_layer: str = 'layer4',
    device: str = 'cuda'
)
```

**Parameters:**
- `model`: Model to explain
- `target_layer`: Name of layer to compute Grad-CAM for
- `device`: Device for inference

### Methods

#### `compute_gradcam(image, target_class=None)`

Compute Grad-CAM heatmap for an image.

**Parameters:**
- `image`: Input tensor (C, H, W) or (1, C, H, W)
- `target_class`: Class to compute for. If None, uses predicted class.

**Returns:** Heatmap array (H, W), values 0-1

#### `compute_attention_metrics(gradcam)`

Compute attention pattern metrics.

**Returns:** Dict with 'entropy', 'concentration', 'peak'

#### `analyze_by_skin_type(test_loader, fitzpatrick_idx=2, n_samples_per_type=10)`

Analyze attention patterns by skin type.

**Returns:** pandas DataFrame

#### `visualize(image, save_path=None)`

Visualize Grad-CAM overlay on image.

**Returns:** matplotlib Figure

---

## UncertaintyEstimator

Monte Carlo Dropout for uncertainty estimation.

### Constructor

```python
UncertaintyEstimator(
    model: nn.Module, 
    n_passes: int = 20,
    device: str = 'cuda'
)
```

**Parameters:**
- `model`: Model to estimate uncertainty for
- `n_passes`: Number of MC forward passes
- `device`: Device for inference

### Methods

#### `estimate(test_loader, fitzpatrick_idx=2)`

Estimate uncertainty for test data.

**Returns:** `UncertaintyResults`

#### `print_report()`

Print uncertainty analysis report.

#### `plot_calibration(save_path=None)`

Plot calibration curve by skin type.

**Returns:** matplotlib Figure

---

## SelectivePredictor

Selective prediction with uncertainty-based deferral.

### Constructor

```python
SelectivePredictor(
    model: nn.Module,
    device: str = 'cuda',
    n_mc_passes: int = 20
)
```

### Methods

#### `evaluate_thresholds(test_loader, thresholds=[0.5, 0.6, 0.7, 0.8, 0.9, 0.95], fitzpatrick_idx=2)`

Evaluate performance at different confidence thresholds.

**Returns:** pandas DataFrame with columns: threshold, coverage, accuracy, improvement, n_retained, n_deferred

#### `predict_with_deferral(test_loader, threshold=0.9, fitzpatrick_idx=2)`

Make predictions with deferral at given threshold.

**Returns:** Tuple (retained_df, deferred_df)

#### `print_report(threshold_df)`

Print selective prediction report.

#### `plot_tradeoff(threshold_df, save_path=None)`

Plot coverage-accuracy tradeoff curve.

**Returns:** matplotlib Figure

---

## MitigationStrategies

Collection of bias mitigation strategies.

### Static Methods

#### `create_balanced_sampler(df, label_col='label', fitzpatrick_col='fitzpatrick')`

Create balanced sampler for training.

**Returns:** WeightedRandomSampler

#### `get_skin_tone_aware_transform(fitzpatrick_type, is_train=True, base_size=224)`

Get transforms with skin-type-aware augmentation.

**Returns:** torchvision.transforms.Compose

#### `create_synthetic_augmented_loader(dataset, batch_size=32, augmentation_prob=0.5, l_shift_range=(-30, 30), **kwargs)`

Create DataLoader with synthetic LAB augmentation.

**Returns:** DataLoader

#### `train_ensemble_specialists(train_df, model_factory, train_fn, fitzpatrick_col='fitzpatrick', groups=None)`

Train ensemble of specialist models.

**Returns:** Dict[str, nn.Module]

#### `ensemble_predict(specialists, images, fitzpatrick_types, device='cuda', groups=None)`

Make predictions using ensemble.

**Returns:** torch.Tensor

---

## Data Classes

### FairnessMetrics

```python
@dataclass
class FairnessMetrics:
    accuracy: float
    sensitivity: float
    specificity: float
    ppv: float
    npv: float
    f1: float
    auc: Optional[float] = None
    n_samples: int = 0
```

### EquityGap

```python
@dataclass
class EquityGap:
    metric: str
    gap: float
    best_group: int
    worst_group: int
    best_value: float
    worst_value: float
```

### AuditResults

```python
@dataclass
class AuditResults:
    metrics_by_fitzpatrick: Dict[int, FairnessMetrics]
    overall_metrics: FairnessMetrics
    equity_gaps: Dict[str, EquityGap]
    predictions_df: pd.DataFrame
```

### CounterfactualResults

```python
@dataclass
class CounterfactualResults:
    flip_rate: float
    flip_rate_by_fst: Dict[int, float]
    avg_prob_range: float
    prob_range_by_fst: Dict[int, float]
    detailed_results: pd.DataFrame
```

### UncertaintyResults

```python
@dataclass
class UncertaintyResults:
    avg_confidence: float
    avg_entropy: float
    confidence_by_fst: Dict[int, float]
    entropy_by_fst: Dict[int, float]
    accuracy_by_fst: Dict[int, float]
    detailed_results: pd.DataFrame
```

---

## Convenience Functions

### `compare_models(models, test_loader, device='cuda')`

Compare multiple models on fairness metrics.

**Parameters:**
- `models`: Dict of model_name -> model
- `test_loader`: Test DataLoader
- `device`: Device for inference

**Returns:** pandas DataFrame

### `quick_audit(model, test_loader, device='cuda')`

Run quick fairness audit and print report.

---

## Example Usage

```python
from dermequity import (
    FairnessAuditor,
    SyntheticLABAugmentation,
    CounterfactualAnalyzer,
    MitigationStrategies,
)

# 1. Audit model
auditor = FairnessAuditor(model, device='cuda')
results = auditor.audit(test_loader)
auditor.print_report()

# 2. Apply best mitigation
train_loader = MitigationStrategies.create_synthetic_augmented_loader(
    train_dataset,
    augmentation_prob=0.5,
    l_shift_range=(-30, 30)
)

# 3. Verify causal bias
analyzer = CounterfactualAnalyzer(model)
cf_results = analyzer.analyze(test_loader)
print(f"Flip rate: {cf_results.flip_rate:.1%}")
```
