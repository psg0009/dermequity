# DermEquity 🩺

**Fairness Auditing & Bias Mitigation Toolkit for Dermatological AI**

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![ShowCAIS 2026](https://img.shields.io/badge/ShowCAIS-2026_Top_Abstract-green.svg)](https://sites.google.com/usc.edu/showcais2026)

DermEquity is a comprehensive framework for **auditing and mitigating skin tone bias** in dermatological AI systems. Built from research presented at ShowCAIS 2026, it provides tools to measure performance disparities across Fitzpatrick skin types and implement evidence-based mitigation strategies.

---

## 🎯 Key Findings

Our research uncovered critical bias patterns:

| Finding | Result |
|---------|--------|
| **Baseline Sensitivity Gap** | 100% (0% on darkest skin) |
| **Bias Scales with Severity** | 25.6% → 75% (benign → malignant) |
| **Counterfactual Flip Rate** | 35.1% predictions change with skin tone |
| **Best Mitigation** | 68% gap reduction (Synthetic LAB) |

---

## 🚀 Features

### Core Analysis
- **Comprehensive Fairness Audit** — Metrics stratified by Fitzpatrick I-VI
- **Equity Gap Calculation** — Max-Min disparity across skin types

### Explainability
- **GradCAM Visualization** — Attention patterns by skin type
- **MC Dropout Uncertainty** — Calibration analysis across groups
- **Counterfactual Analysis** — Causal bias detection via skin tone modification

### Mitigation Strategies
| Strategy | Gap Reduction | Description |
|----------|---------------|-------------|
| Balanced Resampling | ↓19% | Oversample underrepresented groups |
| Skin-Aware Augmentation | ↓41% | Adaptive augmentation by skin type |
| Ensemble of Specialists | ↓56% | Train separate models per group |
| **Synthetic LAB Augmentation** | **↓68%** | L-channel modification (BEST) |
| Selective Prediction | +5.2% acc | Defer low-confidence cases |

---

## 📦 Installation

```bash
# Clone the repository
git clone https://github.com/psg0009/dermequity.git
cd dermequity

# Install dependencies
pip install -r requirements.txt

# Install the package
pip install -e .
```

### Requirements

- Python 3.8+
- PyTorch 1.9+
- torchvision
- OpenCV (cv2)
- scikit-learn
- pandas
- matplotlib
- numpy
- Pillow

---

## 🚀 Quick Start

### 1. Run Complete Analysis

```python
from dermequity import full_analysis

# Load your model and test data
model = load_your_model()
test_loader = create_test_loader()  # yields (images, labels, fitzpatrick_types)

# Run full pipeline
results = full_analysis(model, test_loader, output_dir='./results')
```

### 2. Fairness Audit Only

```python
from dermequity import FairnessAuditor

auditor = FairnessAuditor(model, device='cuda')
results = auditor.audit(test_loader)

auditor.print_report()
auditor.plot_results(save_path='fairness_audit.png')
```

### 3. Counterfactual Analysis (Causal Bias Detection)

```python
from dermequity import CounterfactualAnalyzer

analyzer = CounterfactualAnalyzer(model, device='cuda')
results = analyzer.analyze(test_loader)

print(f"Prediction Flip Rate: {results.flip_rate:.1%}")
# Output: "Prediction Flip Rate: 35.1%"

# Visualize counterfactual variants
analyzer.visualize_counterfactual(image, save_path='counterfactual.png')
```

### 4. Apply Synthetic LAB Augmentation (Best Mitigation)

```python
from dermequity import SyntheticLABTransform
import torchvision.transforms as transforms

# Add to your training transforms
train_transform = transforms.Compose([
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    SyntheticLABTransform(l_shift_range=(-30, 30), p=0.5),  # 🔑 KEY
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])
```

### 5. GradCAM Explainability

```python
from dermequity import GradCAMExplainer

explainer = GradCAMExplainer(model, target_layer='layer4')

# Generate heatmap
heatmap = explainer.generate(image)

# Visualize
explainer.visualize(image, heatmap, save_path='gradcam.png')

# Analyze patterns across skin types
patterns = explainer.analyze_by_fitzpatrick(test_loader)
```

### 6. Uncertainty Quantification (MC Dropout)

```python
from dermequity import UncertaintyEstimator

estimator = UncertaintyEstimator(model, n_forward_passes=20)
results = estimator.estimate(test_loader)

estimator.print_report()
print(f"Calibration Error: {results.calibration_error:.3f}")
```

### 7. Selective Prediction (Deferral)

```python
from dermequity import SelectivePredictor

predictor = SelectivePredictor(model)

# Evaluate thresholds
df = predictor.evaluate_thresholds(test_loader)
print(df)

# At 0.95 threshold: 76% coverage, +5.2% accuracy
predictor.plot_coverage_accuracy_curve(test_loader, save_path='deferral.png')
```

---

## 📊 Output Example

```
================================================================================
DERMEQUITY FAIRNESS AUDIT REPORT
================================================================================

📊 OVERALL PERFORMANCE:
----------------------------------------
   Accuracy:    0.799
   Sensitivity: 0.481
   Specificity: 0.850
   PPV:         0.482
   F1:          0.481

📊 PERFORMANCE BY FITZPATRICK SKIN TYPE:
----------------------------------------------------------------------
Type    N       Acc       Sens      Spec      PPV       AUC       
----------------------------------------------------------------------
1       7       0.571     0.500     0.600     0.333     0.500     
2       21      1.000     1.000     1.000     1.000     1.000     
3       39      0.872     0.571     0.938     0.667     0.821     
4       51      0.686     0.125     0.791     0.100     0.532     
5       46      0.783     0.600     0.805     0.273     0.673     
6       30      0.833     0.000     0.862     0.000     0.379     

🚨 EQUITY GAPS:
----------------------------------------------------------------------
   🔴 HIGH SENSITIVITY: Gap=100.0% (Best: Type 2=1.000, Worst: Type 6=0.000)
   🔴 HIGH ACCURACY: Gap=42.9% (Best: Type 2=1.000, Worst: Type 1=0.571)
   🔴 HIGH PPV: Gap=100.0% (Best: Type 2=1.000, Worst: Type 6=0.000)
================================================================================
```

---

## 🧪 API Reference

### Core Classes

| Class | Purpose |
|-------|---------|
| `FairnessAuditor` | Main fairness audit by Fitzpatrick type |
| `CounterfactualAnalyzer` | Causal bias detection via skin tone modification |
| `GradCAMExplainer` | Attention visualization and analysis |
| `UncertaintyEstimator` | MC Dropout uncertainty quantification |
| `SelectivePredictor` | Deferral-based selective prediction |
| `MitigationStrategies` | Collection of bias mitigation methods |
| `SyntheticLABAugmentation` | LAB color space augmentation |

### Data Classes

| Class | Contents |
|-------|----------|
| `FairnessMetrics` | accuracy, sensitivity, specificity, ppv, npv, f1, auc |
| `EquityGap` | metric, gap, best_group, worst_group, best/worst values |
| `AuditResults` | metrics_by_fitzpatrick, overall_metrics, equity_gaps |
| `CounterfactualResults` | flip_rate, flip_rate_by_fst, avg_prob_change |
| `UncertaintyResults` | avg_confidence, avg_entropy, calibration_error |

---

## 📁 Dataset Compatibility

DermEquity works with any dataset that provides Fitzpatrick skin type labels:

```python
# DataLoader should yield batches of:
# (images, labels, fitzpatrick_types)

for images, labels, fitz_types in test_loader:
    # images: [B, 3, 224, 224] tensor
    # labels: [B] tensor (0 or 1 for binary)
    # fitz_types: [B] list/tensor of Fitzpatrick types (1-6)
```

### Tested Datasets
- [Fitzpatrick17k](https://github.com/mattgroh/fitzpatrick17k) — Primary development dataset
- [Stanford DDI](https://ddi-dataset.github.io/) — External validation (656 biopsy-confirmed)

---

## 📝 Citation

If you use DermEquity in your research, please cite:

```bibtex
@inproceedings{gosar2026dermequity,
  title={DermEquity: Auditing and Mitigating Skin Tone Bias in Dermatological AI Systems},
  author={Gosar, Parth},
  booktitle={ShowCAIS 2026: USC Center for AI in Society Symposium},
  note={Top Abstract Award},
  year={2026}
}
```

### Key References

- Groh et al. "Fitzpatrick 17k Dataset." CVPR Workshop, 2021.
- Daneshjou et al. "Disparities in Dermatology AI: DDI Dataset." Science Advances 8(32), 2022.
- Esteva et al. "Dermatologist-level skin cancer classification." Nature 542, 2017.
- Adamson & Smith. "ML and Health Care Disparities in Dermatology." JAMA Derm, 2018.

---

## 🤝 Contributing

Contributions welcome! Please:

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add AmazingFeature'`)
4. Push to branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

---

## 📄 License

MIT License — see [LICENSE](LICENSE) for details.

---

## 🙏 Acknowledgments

- [Fitzpatrick17k Dataset](https://github.com/mattgroh/fitzpatrick17k) by Matt Groh et al.
- [Stanford DDI Dataset](https://ddi-dataset.github.io/) by Daneshjou et al.
- [USC Center for AI in Society](https://cais.usc.edu/)
- ShowCAIS 2026 organizers and reviewers

---

**Built with ❤️ at USC Viterbi School of Engineering**
