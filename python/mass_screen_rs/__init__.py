"""
The MUSE data-driven response mask (mass screen), in Rust.
"""

from importlib.metadata import version as _get_version

from .screen import mass_screen

__version__ = _get_version("mass-screen-rs")

__all__ = ["__version__", "mass_screen"]
