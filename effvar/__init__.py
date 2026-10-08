"""Efficient VAR: harness, references and cheaper Visual Attention Redistribution for MedGemma."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MODEL_ID = "google/medgemma-1.5-4b-it"
FLAGS = ("retinamnist", "pneumoniamnist", "dermamnist")
