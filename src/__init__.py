"""Shared building blocks for RiskPulse.

Modules
-------
``config``
    Project paths, risk weights and policy thresholds.
``risk``
    Pure risk math: loan-to-income, policy grade, blend and advice.
``features``
    Validation and transformation of applicant inputs into model features.
``artifacts``
    Loaders for the trained model, the scaler and the reference metrics.
"""

from src.config import LTI_WEIGHT, MODEL_WEIGHT

__all__ = ["LTI_WEIGHT", "MODEL_WEIGHT"]
