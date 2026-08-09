#!/usr/bin/env python
"""Wrapper so torchrun can spawn src.evaluation.eval as a module."""
import os
import sys

# Make 'src' importable
sys.path.insert(0, "/data2/fzx/SERUM")

# Re-execute src.evaluation.eval:sys.argv as a real module
import runpy
runpy.run_module("src.evaluation.eval", run_name="__main__", alter_sys=True)