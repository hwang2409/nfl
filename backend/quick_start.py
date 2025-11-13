#!/usr/bin/env python3
"""
Quick Start Script for NFL Prediction System

This script runs through the entire pipeline:
1. Collect data
2. Train improved model
3. Make predictions
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from data_collection import collect_game_data
from train_improved_model import main as train_model
from predict_upcoming_improved import predict_upcoming_improved

def main():
    print("=" * 60)
    print("NFL Game Prediction System - Quick Start")
    print("=" * 60)
    
    # Step 1: Collect data
    print("\n[1/3] Collecting NFL data...")
    print("Note: This may take a few minutes for multiple seasons")
    print("Using most recent data for better prediction accuracy")
    # Use recent data: last 6-7 seasons (defaults handled in collect_game_data)
    pbp, games = collect_game_data()
    
    if pbp is None:
        print("Error: Failed to collect data")
        return
    
    # Step 2: Train improved model
    print("\n[2/3] Training improved model...")
    print("This will prepare features and train the model.")
    train_model()
    
    # Step 3: Make predictions
    print("\n[3/3] Making predictions for upcoming games...")
    predictions = predict_upcoming_improved()
    
    if predictions is not None and len(predictions) > 0:
        print(f"\n✓ Generated {len(predictions)} predictions")
    else:
        print("\n⚠️  No predictions generated")
    
    print("\n" + "=" * 60)
    print("Setup complete! You can now make predictions.")
    print("=" * 60)
    print("\nExample commands:")
    print("  # Make predictions for specific week")
    print("  python src/predict_upcoming_improved.py --season 2024 --week 1")
    print("\n  # Evaluate model")
    print("  python src/evaluate_model.py --season 2023")

if __name__ == "__main__":
    main()

