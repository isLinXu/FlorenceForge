#!/usr/bin/env python3
"""Legacy setuptools shim.

FlorenceForge uses `pyproject.toml` as the single source of truth for package
metadata, dependencies, extras, and console scripts. This file remains only for
older tooling that still invokes `setup.py` directly.
"""

from setuptools import setup


if __name__ == "__main__":
    setup()
