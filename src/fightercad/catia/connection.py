"""CATIA COM automation connection manager.

Provides live connection to CATIA V5/V6 via Windows COM API (pywin32).
Creates and updates parts directly inside CATIA's specification tree.
"""

from __future__ import annotations

import sys
from typing import Any

_HAS_WIN32 = False
_CATIA = None

if sys.platform == "win32":
    try:
        import win32com.client
        _HAS_WIN32 = True
    except ImportError:
        pass


def is_catia_available() -> bool:
    """Check if CATIA COM connection is possible (Windows + pywin32)."""
    return _HAS_WIN32


def connect_catia() -> Any:
    """Connect to a running CATIA instance.

    Returns the CATIA application COM object.
    Raises RuntimeError if CATIA is not running or not on Windows.
    """
    global _CATIA
    if not _HAS_WIN32:
        raise RuntimeError(
            "CATIA COM requires Windows and pywin32. "
            "Install pywin32: pip install pywin32"
        )
    try:
        _CATIA = win32com.client.GetActiveObject("CATIA.Application")
        return _CATIA
    except Exception as e:
        raise RuntimeError(
            f"Could not connect to CATIA. Make sure CATIA is running. Error: {e}"
        )


def get_catia() -> Any:
    """Get the current CATIA COM connection, connecting if needed."""
    global _CATIA
    if _CATIA is None:
        return connect_catia()
    return _CATIA


def disconnect():
    """Release the CATIA COM connection."""
    global _CATIA
    _CATIA = None
