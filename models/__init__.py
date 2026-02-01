"""Price forecasting models."""


def check_package(name: str) -> bool:
    """Check if an optional package is installed."""
    try:
        __import__(name)
        return True
    except ImportError:
        return False
