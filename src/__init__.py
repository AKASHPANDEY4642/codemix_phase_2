"""
MedManglish-RAG: A Script-Aware Retrieval-Augmented Generation Framework
for Marathi-English Code-Mixed Medical QA.

Master's Thesis Project.
"""

__version__ = "2.0.0"
__author__ = "Akash Pandey"

import os
import yaml
import logging
from pathlib import Path
from typing import Any, Dict
from dotenv import load_dotenv

# ---------------------------------------------------------------------------
# Load environment variables from .env
# ---------------------------------------------------------------------------
_PROJECT_ROOT = Path(__file__).resolve().parents[1]
load_dotenv(_PROJECT_ROOT / ".env")
load_dotenv(_PROJECT_ROOT / "config" / ".env.example")  # Fallback defaults

# ---------------------------------------------------------------------------
# Config loader
# ---------------------------------------------------------------------------
_CONFIG_CACHE: Dict[str, Any] = {}


def get_config() -> Dict[str, Any]:
    """Load and cache the centralized YAML configuration.

    Returns:
        Dict[str, Any]: The full configuration dictionary.
    """
    if _CONFIG_CACHE:
        return _CONFIG_CACHE

    config_path = _PROJECT_ROOT / "config" / "settings.yaml"
    if not config_path.exists():
        raise FileNotFoundError(
            f"Configuration file not found: {config_path}. "
            "Copy config/settings.yaml.example to config/settings.yaml."
        )

    with open(config_path, "r", encoding="utf-8") as fh:
        config = yaml.safe_load(fh)

    _CONFIG_CACHE.update(config)
    return _CONFIG_CACHE


def get_project_root() -> Path:
    """Return the absolute project root path.

    Returns:
        Path: Absolute path to the project root directory.
    """
    return _PROJECT_ROOT


# ---------------------------------------------------------------------------
# Logging setup
# ---------------------------------------------------------------------------
def setup_logging(name: str = "medmanglish") -> logging.Logger:
    """Create a configured logger instance.

    Args:
        name: Logger name (typically the module name).

    Returns:
        logging.Logger: Configured logger.
    """
    config = get_config()
    log_cfg = config.get("logging", {})

    logger = logging.getLogger(name)
    if logger.handlers:
        return logger  # Already configured

    level = getattr(logging, log_cfg.get("level", "INFO").upper(), logging.INFO)
    logger.setLevel(level)

    fmt = logging.Formatter(
        log_cfg.get("format", "%(asctime)s | %(name)s | %(levelname)s | %(message)s")
    )

    # Console handler
    ch = logging.StreamHandler()
    ch.setLevel(level)
    ch.setFormatter(fmt)
    logger.addHandler(ch)

    # File handler (optional)
    log_file = log_cfg.get("file")
    if log_file:
        log_path = _PROJECT_ROOT / log_file
        log_path.parent.mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(str(log_path), encoding="utf-8")
        fh.setLevel(level)
        fh.setFormatter(fmt)
        logger.addHandler(fh)

    return logger
