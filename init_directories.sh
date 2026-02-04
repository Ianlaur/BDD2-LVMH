#!/bin/bash
# Initialize directory structure for LVMH server

cd /Users/ian/BDD2-LVMH

echo "Creating directory structure..."

mkdir -p data/input
mkdir -p data/processed
mkdir -p data/outputs
mkdir -p taxonomy
mkdir -p models/sentence_transformers
mkdir -p activations
mkdir -p nltk_data

echo "✓ All directories created"
echo ""
echo "Directory structure:"
ls -la data/
