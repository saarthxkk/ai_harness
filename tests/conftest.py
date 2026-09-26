"""Pytest configuration — adds the project root to sys.path."""

import sys
import pathlib

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
