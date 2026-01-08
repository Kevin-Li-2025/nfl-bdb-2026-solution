#!/usr/bin/env python3
"""
Quick syntax verification test.
Run this to verify the code is syntactically correct.
"""

# Test 1: Import check (no external deps)
print("Testing syntax...")

import ast
import sys
from pathlib import Path

def check_syntax(filepath):
    """Check Python file syntax."""
    try:
        with open(filepath, 'r') as f:
            source = f.read()
        ast.parse(source)
        print(f"✓ {filepath}: Syntax OK")
        return True
    except SyntaxError as e:
        print(f"✗ {filepath}: {e}")
        return False

# Check all Python files
src_dir = Path(__file__).parent / "src"
files = [
    src_dir / "model.py",
    src_dir / "dataset.py",
    src_dir / "train.py",
    src_dir / "inference.py",
    Path(__file__).parent / "kaggle_notebook.py"
]

all_ok = True
for f in files:
    if f.exists():
        if not check_syntax(f):
            all_ok = False
    else:
        print(f"⚠ {f}: File not found")

if all_ok:
    print("\n" + "="*50)
    print("✓ All files have valid syntax!")
    print("="*50)
    print("\nTo run on Kaggle:")
    print("1. Upload kaggle_notebook.py to a Kaggle notebook")
    print("2. Or create a new notebook and copy nfl_bdb_2026_solution.ipynb")
    print("3. Run with GPU acceleration enabled")
else:
    print("\n✗ Some files have syntax errors")
    sys.exit(1)
