"""Pytest configuration and shared fixtures."""

import pytest


# Configure pytest-asyncio mode
pytest_plugins = ('pytest_asyncio',)


def pytest_configure(config):
    """Configure pytest-asyncio to auto mode."""
    config.addinivalue_line(
        "markers", "asyncio: mark test as async"
    )
