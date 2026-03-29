# DermEquity Training Notebooks

## 📓 Notebooks

### `DermEquity_Complete_Training.ipynb`

Complete training and evaluation notebook for reproducing all experiments.

**Requirements:**
- Google Colab with GPU (T4 or better)
- ~60-75 minutes runtime

**Sections:**

| Cells | Section | Description |
|-------|---------|-------------|
| 1-10 | **Setup & Data** | Install packages, load Fitzpatrick17k, download images |
| 11-16 | **Baseline Training** | Train EfficientNet-B0 binary classifier |
| 17 | **Baseline Evaluation** | Evaluate by Fitzpatrick type, compute equity gaps |
| 18-19 | **Strategy 1: Balanced Resampling** | Weighted sampling by skin type |
| 26-27 | **Strategy 2: Skin-Aware Augmentation** | Type-specific augmentation |
| 28-29 | **Strategy 3: Ensemble** | Specialist models per skin type group |
| 33-39 | **Multi-Class Analysis** | Equity gaps by condition severity |
| 40-44 | **Grad-CAM Explainability** | Attention patterns by skin type |
| 45-49 | **Uncertainty Estimation** | MC Dropout analysis |
| 50-54 | **Strategy 4: Synthetic LAB Augmentation** | L-channel color space augmentation |
| 55-58 | **Counterfactual Analysis** | Causal bias detection (35.1% flip rate) |
| 59-63 | **Cross-Dataset Validation** | Evaluate on Stanford DDI dataset |

---

## 📊 Results

### `dermequity_complete_results.json`

Complete experimental results in JSON format.

**Contents:**
- `mitigation_strategies`: All 5 strategies with accuracy, sensitivity, specificity, AUC
- `multiclass_analysis`: Gaps by condition severity
- `cross_dataset_ddi`: DDI validation results
- `key_findings`: Summary statistics

---

## 🚀 Quick Start

1. Open `DermEquity_Complete_Training.ipynb` in Google Colab
2. **Enable GPU**: Runtime → Change runtime type → GPU
3. Run all cells (Runtime → Run all)
4. Results will be saved automatically

---

## 📈 Key Results

| Metric | Baseline | Best Mitigation | Improvement |
|--------|----------|-----------------|-------------|
| Accuracy Gap | 42.9% | 13.8% | ↓68% |
| Sensitivity Gap | 100% | 75% | ↓25% |
| AUC Gap | 62.1% | 41.4% | ↓33% |

---

## 📚 Citation

```bibtex
@software{gosar2026dermequity,
  author = {Gosar, Parth},
  title = {DermEquity: Fairness Auditing Toolkit for Dermatological AI},
  year = {2026},
  publisher = {GitHub},
  url = {https://github.com/psg0009/dermequity}
}
```
