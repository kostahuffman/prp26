"""
Test suite for the prp26 package.
"""

import pytest


def test_package_import():
    """Test that the package can be imported."""
    import prp26

    assert hasattr(prp26, "__version__")
    assert prp26.__version__ == "0.1.0"


def test_subpackage_imports():
    """Test that subpackages can be imported."""
    from prp26 import core, models, numerics, products

    # These should import without error
    assert core is not None
    assert products is not None
    assert models is not None
    assert numerics is not None


def test_models_submodules():
    """Test that model submodules can be imported."""
    from prp26.models import correlation, dividends, rates, volatility

    assert volatility is not None
    assert correlation is not None
    assert rates is not None
    assert dividends is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
