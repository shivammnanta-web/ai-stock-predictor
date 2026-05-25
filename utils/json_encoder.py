"""
Custom JSON encoder for handling numpy types gracefully.
Used across Flask app and index.py for consistent serialization.
"""
import json
import numpy as np


class NumpyEncoder(json.JSONEncoder):
    """Handles numpy int/float/array serialization for JSON responses."""
    def default(self, obj):
        if isinstance(obj, (np.integer, np.int64)):
            return int(obj)
        if isinstance(obj, (np.floating, np.float64)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)
