# src/config/__init__.py
# Configuration package for loading and managing application settings.

from src.config.settings import get_settings, Settings

__all__ = ["get_settings", "Settings"]
