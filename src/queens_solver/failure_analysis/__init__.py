"""Failure analysis and model comparison tools for Queens puzzle solvers."""

from queens_solver.failure_analysis.analyzer import (
    FailureAnalyzer,
    MODEL_CONFIGS,
    evaluate_models_on_dataset,
    visualize_failure_statistics,
    visualize_model_comparison,
)

__all__ = [
    'FailureAnalyzer',
    'MODEL_CONFIGS',
    'evaluate_models_on_dataset',
    'visualize_failure_statistics',
    'visualize_model_comparison',
]
