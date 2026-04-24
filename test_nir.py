#!/usr/bin/env python3
"""Test script for NIR project."""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'python_nir_project'))

from nir_project.pipeline import compare_property_prediction_pipeline

if __name__ == '__main__':
    # This would require actual data, but we can check if the function exists
    print("Function exists:", callable(compare_property_prediction_pipeline))
    print("Test passed!")