#!/bin/bash
# Script to run the Swiss Knife knockout tournament

# Activate conda environment
source activate myenv || conda activate myenv || echo "Please activate the myenv conda environment manually"

echo "=========================================================="
echo "Running Swiss Knife Knockout Tournament - Helpfulness Blade"
echo "=========================================================="
# This will download the base model and the helpfulness adapter, then run the generation
python -m Model_mechanics.main \
    --prompt "Explain the concept of AI alignment to a beginner." \
    --blade helpfulness \
    --alpha 0.5 \
    --K 8 \
    --L 5 \
    --max-tokens 50 \
    --verbose

echo "\n\n=========================================================="
echo "Running Swiss Knife Knockout Tournament - Truthfulness Blade"
echo "=========================================================="
# This will download the truthfulness adapter (base model is already cached), then run the generation
python -m Model_mechanics.main \
    --prompt "Is the earth flat or round?" \
    --blade truthfulness \
    --alpha 0.5 \
    --K 8 \
    --L 5 \
    --max-tokens 50 \
    --verbose

echo "\nDone."
