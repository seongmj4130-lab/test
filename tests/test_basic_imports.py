"""
Basic import smoke tests for the quant trading system.

This module tests that all core modules can be imported without errors.
"""
import sys
import os

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))


def test_core_dependencies():
    """Test that core data science dependencies can be imported."""
    try:
        import numpy as np
        import pandas as pd
        import sklearn
        import xgboost
        import lightgbm
        assert True, "Core dependencies imported successfully"
    except ImportError as e:
        assert False, f"Failed to import core dependencies: {e}"


def test_project_modules():
    """Test that project modules can be imported."""
    modules_to_test = [
        'src.core',
        'src.utils.config',
        'src.utils.io',
        'src.components.ranking.score_engine',
        'src.components.portfolio.selector',
    ]

    failed_imports = []

    for module in modules_to_test:
        try:
            __import__(module)
        except ImportError as e:
            failed_imports.append(f"{module}: {e}")
        except Exception as e:
            failed_imports.append(f"{module}: {e}")

    if failed_imports:
        failure_msg = "Failed to import modules:\n" + "\n".join(failed_imports)
        assert False, failure_msg


def test_config_loading():
    """Test that configuration can be loaded."""
    try:
        import yaml
        import os

        config_path = os.path.join(os.path.dirname(__file__), '..', 'configs', 'config.yaml')
        if os.path.exists(config_path):
            with open(config_path, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
            assert isinstance(config, dict), "Config should be a dictionary"
            assert 'l4' in config, "Config should contain l4 section"
        else:
            # Config file doesn't exist, but that's okay for basic import test
            assert True, "Config file not found, but import test passed"
    except Exception as e:
        assert False, f"Config loading failed: {e}"


def test_basic_numpy_pandas():
    """Test basic numpy and pandas operations."""
    try:
        import numpy as np
        import pandas as pd

        # Test numpy
        arr = np.array([1, 2, 3, 4, 5])
        assert arr.sum() == 15, "NumPy sum failed"

        # Test pandas
        df = pd.DataFrame({'a': [1, 2, 3], 'b': [4, 5, 6]})
        assert df.shape == (3, 2), "Pandas DataFrame shape incorrect"
        assert df['a'].sum() == 6, "Pandas sum failed"

    except Exception as e:
        assert False, f"Basic numpy/pandas operations failed: {e}"


def test_python_version():
    """Test that we're running a supported Python version."""
    import sys

    version = sys.version_info
    assert version.major == 3, f"Python major version should be 3, got {version.major}"
    assert version.minor >= 13, f"Python minor version should be >= 13, got {version.minor}"

    # Test that we can import typing features available in Python 3.13+
    try:
        from typing import TypeVar
        T = TypeVar('T')
        assert True, "TypeVar import successful"
    except ImportError:
        assert False, "Failed to import TypeVar"


if __name__ == "__main__":
    # Allow running this test directly
    test_core_dependencies()
    test_project_modules()
    test_config_loading()
    test_basic_numpy_pandas()
    test_python_version()
    print("All basic import tests passed!")