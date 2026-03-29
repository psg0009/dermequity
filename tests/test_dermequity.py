"""
DermEquity Test Suite
=====================

Unit tests for the DermEquity fairness auditing toolkit.
"""

import pytest
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

# Import DermEquity components
from dermequity import (
    FairnessMetrics,
    EquityGap,
    BiasMetrics,
    FairnessAuditor,
    SyntheticLABAugmentation,
    CounterfactualAnalyzer,
    UncertaintyEstimator,
    SelectivePredictor,
    MitigationStrategies,
)


# =============================================================================
# FIXTURES
# =============================================================================

class MockDataset(Dataset):
    """Mock dataset for testing."""
    
    def __init__(self, n_samples=100):
        self.n_samples = n_samples
        np.random.seed(42)
        self.labels = np.random.randint(0, 2, n_samples)
        self.fitzpatrick = np.random.randint(1, 7, n_samples)
        
    def __len__(self):
        return self.n_samples
    
    def __getitem__(self, idx):
        image = torch.randn(3, 224, 224)
        return image, self.labels[idx], self.fitzpatrick[idx]


class MockModel(nn.Module):
    """Simple mock model for testing."""
    
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(3 * 224 * 224, 2)
        self.dropout = nn.Dropout(0.5)
    
    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = self.dropout(x)
        return self.fc(x)


@pytest.fixture
def mock_model():
    return MockModel()


@pytest.fixture
def mock_dataset():
    return MockDataset(n_samples=50)


@pytest.fixture
def mock_loader(mock_dataset):
    return DataLoader(mock_dataset, batch_size=8, shuffle=False)


# =============================================================================
# TEST: DATA CLASSES
# =============================================================================

class TestDataClasses:
    """Test data class functionality."""
    
    def test_fairness_metrics_creation(self):
        metrics = FairnessMetrics(
            accuracy=0.85,
            sensitivity=0.80,
            specificity=0.90,
            ppv=0.75,
            npv=0.92,
            f1=0.77,
            auc=0.88,
            n_samples=100
        )
        
        assert metrics.accuracy == 0.85
        assert metrics.sensitivity == 0.80
        assert metrics.n_samples == 100
    
    def test_equity_gap_str(self):
        gap = EquityGap(
            metric='sensitivity',
            gap=0.5,
            best_group=2,
            worst_group=6,
            best_value=1.0,
            worst_value=0.5
        )
        
        gap_str = str(gap)
        assert 'SENSITIVITY' in gap_str
        assert '50.0%' in gap_str


# =============================================================================
# TEST: BIAS METRICS
# =============================================================================

class TestBiasMetrics:
    """Test bias metrics calculations."""
    
    def test_compute_specificity(self):
        y_true = np.array([0, 0, 0, 1, 1])
        y_pred = np.array([0, 0, 1, 1, 1])
        
        spec = BiasMetrics.compute_specificity(y_true, y_pred)
        assert 0 <= spec <= 1
        assert spec == 2/3  # 2 TN out of 3 negatives
    
    def test_compute_npv(self):
        y_true = np.array([0, 0, 0, 1, 1])
        y_pred = np.array([0, 0, 1, 1, 1])
        
        npv = BiasMetrics.compute_npv(y_true, y_pred)
        assert 0 <= npv <= 1
    
    def test_compute_all_metrics(self):
        y_true = np.array([0, 0, 1, 1, 1, 0, 1, 0])
        y_pred = np.array([0, 1, 1, 1, 0, 0, 1, 0])
        y_prob = np.array([0.2, 0.6, 0.8, 0.9, 0.4, 0.3, 0.7, 0.1])
        
        metrics = BiasMetrics.compute_all_metrics(y_true, y_pred, y_prob)
        
        assert isinstance(metrics, FairnessMetrics)
        assert 0 <= metrics.accuracy <= 1
        assert 0 <= metrics.sensitivity <= 1
        assert 0 <= metrics.specificity <= 1
        assert metrics.n_samples == 8
    
    def test_empty_array(self):
        y_true = np.array([])
        y_pred = np.array([])
        
        metrics = BiasMetrics.compute_all_metrics(y_true, y_pred)
        assert metrics.n_samples == 0


# =============================================================================
# TEST: FAIRNESS AUDITOR
# =============================================================================

class TestFairnessAuditor:
    """Test fairness auditor functionality."""
    
    def test_auditor_initialization(self, mock_model):
        auditor = FairnessAuditor(mock_model, device='cpu')
        assert auditor.model is not None
        assert auditor.results is None
    
    def test_audit_runs(self, mock_model, mock_loader):
        auditor = FairnessAuditor(mock_model, device='cpu')
        results = auditor.audit(mock_loader)
        
        assert results is not None
        assert hasattr(results, 'metrics_by_fitzpatrick')
        assert hasattr(results, 'overall_metrics')
        assert hasattr(results, 'equity_gaps')
    
    def test_audit_results_structure(self, mock_model, mock_loader):
        auditor = FairnessAuditor(mock_model, device='cpu')
        results = auditor.audit(mock_loader)
        
        # Check metrics by fitzpatrick
        assert len(results.metrics_by_fitzpatrick) > 0
        
        # Check equity gaps
        assert 'accuracy' in results.equity_gaps
        assert 'sensitivity' in results.equity_gaps
    
    def test_to_dict(self, mock_model, mock_loader):
        auditor = FairnessAuditor(mock_model, device='cpu')
        auditor.audit(mock_loader)
        
        result_dict = auditor.to_dict()
        
        assert 'overall' in result_dict
        assert 'by_fitzpatrick' in result_dict
        assert 'equity_gaps' in result_dict


# =============================================================================
# TEST: SYNTHETIC LAB AUGMENTATION
# =============================================================================

class TestSyntheticLABAugmentation:
    """Test synthetic LAB augmentation."""
    
    def test_augmentation_initialization(self):
        aug = SyntheticLABAugmentation(l_shift_range=(-30, 30))
        assert aug.l_shift_range == (-30, 30)
    
    def test_augment_single_image(self):
        aug = SyntheticLABAugmentation()
        image = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
        
        augmented = aug.augment(image, l_shift=20)
        
        assert augmented.shape == image.shape
        assert augmented.dtype == np.uint8
    
    def test_augment_with_negative_shift(self):
        aug = SyntheticLABAugmentation()
        image = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
        
        augmented = aug.augment(image, l_shift=-20)
        
        assert augmented.shape == image.shape
    
    def test_generate_variants(self):
        """Test generate_variants method - returns List[np.ndarray]"""
        aug = SyntheticLABAugmentation()
        image = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
        
        variants = aug.generate_variants(image, n_variants=5)
        
        assert len(variants) == 5
        # Each variant is just an image, not a tuple
        for img in variants:
            assert isinstance(img, np.ndarray)
            assert img.shape == image.shape


# =============================================================================
# TEST: MITIGATION STRATEGIES
# =============================================================================

class TestMitigationStrategies:
    """Test mitigation strategies."""
    
    def test_create_balanced_sampler(self):
        df = pd.DataFrame({
            'label': [0, 0, 0, 1, 1, 1, 0, 1],
            'fitzpatrick': [1, 1, 2, 2, 3, 3, 4, 4]
        })
        
        sampler = MitigationStrategies.create_balanced_sampler(
            df, 
            label_col='label',
            fitzpatrick_col='fitzpatrick'
        )
        
        assert sampler is not None
        assert len(sampler) == len(df)
    
    def test_skin_tone_aware_transform(self):
        transform = MitigationStrategies.get_skin_tone_aware_transform(
            fitzpatrick_type=5,
            is_train=True
        )
        
        assert transform is not None


# =============================================================================
# TEST: UNCERTAINTY ESTIMATOR
# =============================================================================

class TestUncertaintyEstimator:
    """Test MC Dropout uncertainty estimation."""
    
    def test_estimator_initialization(self, mock_model):
        """Test initialization with n_forward_passes parameter"""
        estimator = UncertaintyEstimator(mock_model, n_forward_passes=5, device='cpu')
        assert estimator.n_forward_passes == 5
    
    def test_estimate_runs(self, mock_model, mock_loader):
        """Test estimate method runs successfully"""
        estimator = UncertaintyEstimator(mock_model, n_forward_passes=3, device='cpu')
        results = estimator.estimate(mock_loader)
        
        assert results is not None
        assert hasattr(results, 'avg_confidence')
        assert hasattr(results, 'avg_entropy')
        assert 0 <= results.avg_confidence <= 1


# =============================================================================
# TEST: SELECTIVE PREDICTOR
# =============================================================================

class TestSelectivePredictor:
    """Test selective prediction with deferral."""
    
    def test_predictor_initialization(self, mock_model):
        """Test basic initialization"""
        predictor = SelectivePredictor(mock_model, device='cpu')
        assert predictor is not None
    
    def test_predictor_with_uncertainty_estimator(self, mock_model):
        """Test initialization with uncertainty estimator"""
        estimator = UncertaintyEstimator(mock_model, n_forward_passes=3, device='cpu')
        predictor = SelectivePredictor(mock_model, uncertainty_estimator=estimator, device='cpu')
        assert predictor is not None
        assert predictor.uncertainty_estimator is not None
    
    def test_evaluate_thresholds(self, mock_model, mock_loader):
        """Test threshold evaluation - may return fewer rows if some thresholds have no samples"""
        predictor = SelectivePredictor(mock_model, device='cpu')
        
        threshold_df = predictor.evaluate_thresholds(
            mock_loader,
            thresholds=[0.5, 0.7, 0.9]
        )
        
        # Should return at least 1 row, but maybe not all thresholds have samples
        assert len(threshold_df) >= 1
        assert 'threshold' in threshold_df.columns
        assert 'coverage' in threshold_df.columns
        assert 'accuracy' in threshold_df.columns


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
