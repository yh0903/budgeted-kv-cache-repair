"""Budgeted KV-cache repair with semantic-unit selection and packed execution."""

from .api import RepairResult, Selection, repair_after_deletion, select_support

__all__ = ["RepairResult", "Selection", "repair_after_deletion", "select_support"]
__version__ = "0.1.0"
