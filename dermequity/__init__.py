"""
DermEquity: Fairness Auditing Toolkit for Dermatological AI
============================================================

A comprehensive framework for auditing and mitigating skin tone bias
in dermatological AI systems.

Author: Parth Gosar
Affiliation: Thomas Lord Department of Computer Science, USC
License: MIT
Version: 2.0.0
"""

from .dermequity import (
    # Core data classes
    FairnessMetrics,
    EquityGap,
    AuditResults,
    CounterfactualResults,
    UncertaintyResults,
    
    # Metrics
    BiasMetrics,
    
    # Core auditing
    FairnessAuditor,
    
    # Mitigation strategies
    MitigationStrategies,
    SyntheticLABAugmentation,
    SyntheticLABTransform,
    
    # Explainability
    CounterfactualAnalyzer,
    GradCAMExplainer,
    
    # Uncertainty
    UncertaintyEstimator,
    
    # Selective prediction
    SelectivePredictor,
    
    # Convenience functions
    compare_models,
    full_analysis,
)

__version__ = '2.0.0'
__author__ = 'Parth Gosar'
__affiliation__ = 'Thomas Lord Department of Computer Science, USC'

__all__ = [
    # Data classes
    'FairnessMetrics',
    'EquityGap',
    'AuditResults',
    'CounterfactualResults',
    'UncertaintyResults',
    
    # Metrics
    'BiasMetrics',
    
    # Core
    'FairnessAuditor',
    
    # Mitigation
    'MitigationStrategies',
    'SyntheticLABAugmentation',
    'SyntheticLABTransform',
    
    # Explainability
    'CounterfactualAnalyzer',
    'GradCAMExplainer',
    
    # Uncertainty
    'UncertaintyEstimator',
    
    # Selective prediction
    'SelectivePredictor',
    
    # Functions
    'compare_models',
    'full_analysis',
]
