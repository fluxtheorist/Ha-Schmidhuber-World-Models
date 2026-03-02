#!/bin/bash

# Reorganize World Models repo
# Run this from your project root directory

echo "=== Reorganizing World Models repo ==="

# Create new directory structure
echo "Creating directories..."
mkdir -p models
mkdir -p scripts
mkdir -p outputs/iter0
mkdir -p outputs/iter1
mkdir -p experiments

# Move model files
echo "Moving model files..."
mv vae.py models/
mv mdn_rnn.py models/
mv controller.py models/

# Move training/collection scripts
echo "Moving scripts..."
mv collect_data.py scripts/
mv collect_with_controller.py scripts/
mv train_vae.py scripts/
mv train_mdn.py scripts/
mv train_controller.py scripts/
mv watch_controller.py scripts/watch.py

# Move experimental/dream stuff
echo "Moving experimental files..."
mv dream_env.py experiments/
mv reward_predictor.py experiments/
mv train_reward_predictor.py experiments/
mv train_controller_dream.py experiments/
mv simple_autoencoder.py experiments/
mv test_dream_controller.py experiments/
mv test_dream_speed.py experiments/

# Organize outputs into iterations
echo "Organizing outputs..."
# Iteration 0 files
mv outputs/frames.npy outputs/iter0/ 2>/dev/null
mv outputs/actions.npy outputs/iter0/ 2>/dev/null
mv outputs/vae.pth outputs/iter0/ 2>/dev/null
mv outputs/mdn_rnn.pth outputs/iter0/ 2>/dev/null
mv outputs/controller_params.npy outputs/iter0/ 2>/dev/null
mv outputs/controller.pth outputs/iter0/ 2>/dev/null

# Iteration 1 files
mv outputs/frames_iter1.npy outputs/iter1/frames.npy 2>/dev/null
mv outputs/actions_iter1.npy outputs/iter1/actions.npy 2>/dev/null
mv outputs/vae_iter1.pth outputs/iter1/vae.pth 2>/dev/null
mv outputs/mdn_rnn_iter1.pth outputs/iter1/mdn_rnn.pth 2>/dev/null
mv outputs/controller_params_iter1.npy outputs/iter1/controller_params.npy 2>/dev/null

# Delete test files and redundant iter1 training scripts
echo "Removing test files..."
rm -f test_env.py
rm -f test_vae.py
rm -f test_mdn.py
rm -f test_controller.py
rm -f test_saved_controller.py
rm -f controller_confirm.py
rm -f train_vae_iter1.py
rm -f train_mdn_iter1.py
rm -f train_controller_iter1.py
rm -f watch_iter1.py

echo ""
echo "=== Done! ==="
echo ""
echo "New structure:"
echo "  models/        - vae.py, mdn_rnn.py, controller.py"
echo "  scripts/       - training and data collection scripts"
echo "  outputs/iter0/ - iteration 0 data and models"
echo "  outputs/iter1/ - iteration 1 data and models"
echo "  experiments/   - dream training and other experiments"
echo ""
echo "NEXT STEP: You need to update imports in the scripts."
echo "Run: python scripts/train_vae.py  (it will fail with import error)"
echo "Then we'll fix the imports together."
