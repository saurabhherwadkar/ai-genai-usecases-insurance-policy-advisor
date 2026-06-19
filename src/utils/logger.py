"""
Centralized logging setup for the Insurance Policy Advisor application.

Configures logging from config/logging.yaml and provides a factory
function for obtaining module-specific loggers. Supports configurable
log levels via settings.
"""

import logging
import logging.config
import os
from pathlib import Path

import yaml

from src.config.settings import get_settings


def _setup_logging() -> None:
    """
    Initialize the logging system from the logging.yaml configuration file.

    Creates the logs directory if it does not exist, then loads and applies
    the logging configuration. Falls back to basic config if the YAML file
    is not found.
    """
    # Ensure the logs directory exists for file handler
    logs_dir = Path("logs")
    logs_dir.mkdir(exist_ok=True)

    # Locate the logging configuration file
    config_path = Path(__file__).parent.parent.parent / "config" / "logging.yaml"

    if config_path.exists():
        # Load logging config from YAML file
        with open(config_path, "r", encoding="utf-8") as log_config_file:
            log_config = yaml.safe_load(log_config_file)

        # Apply the configured log level from application settings
        settings = get_settings()
        log_level = settings.app.log_level.upper()

        # Override the root logger level with the configured level
        if "root" in log_config:
            log_config["root"]["level"] = log_level

        # Override the src logger level with the configured level
        if "loggers" in log_config and "src" in log_config["loggers"]:
            log_config["loggers"]["src"]["level"] = log_level

        # Apply the logging configuration
        logging.config.dictConfig(log_config)
    else:
        # Fall back to basic logging configuration
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        )


# Initialize logging when this module is first imported
_setup_logging()


def get_logger(name: str) -> logging.Logger:
    """
    Get a logger instance for the specified module name.

    Provides a consistent way to obtain loggers throughout the application.
    All loggers inherit the configuration set up during module initialization.

    Args:
        name: The name for the logger, typically __name__ of the calling module.

    Returns:
        A configured Logger instance for the given name.
    """
    return logging.getLogger(name)
