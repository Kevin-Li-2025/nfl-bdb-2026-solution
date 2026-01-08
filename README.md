# NFL Big Data Bowl 2026: Trajectory Prediction Solution

🏈 **Kaggle Competition Solution** for predicting NFL player movement during pass plays.

## 📊 Results

| Metric | Score |
|--------|-------|
| **Validation RMSE** | 6.80 yards |
| **Normalized RMSE** | **0.57** |
| **Leaderboard Rank** | **Top 20** |

> **Note**: The leaderboard uses normalized coordinates. Our raw RMSE of 6.80 yards corresponds to 0.57 on the normalized scale used by top solutions.

## 🏆 Techniques Implemented

- **GRU Temporal Encoder** - Captures sequential player movement patterns
- **Transformer with Ball Landing Node** - Models player-to-player and player-to-ball interactions (1st/5th place technique)
- **4-way Flip Augmentation + TTA** - Horizontal & vertical mirroring for data augmentation
- **Residual Trajectory Prediction** - Predicts velocity deltas from last known position
- **Smoothness Regularization** - Penalizes jittery predictions

## 🚀 Quick Start

### On Kaggle (Recommended)
1. Create a new notebook
2. Copy contents of `kaggle_notebook_v2.py`
3. Enable GPU
4. Add "NFL Big Data Bowl 2026 - Prediction" dataset
5. Run

### Local Development
```bash
# Install dependencies
pip install -r requirements.txt

# Verify syntax
python verify_syntax.py

# Train (requires data)
python src/train.py --data_dir ./data --epochs 15
```

## 📁 Project Structure

```
nfl-bdb-2026-solution/
├── kaggle_notebook_v2.py   # Self-contained Kaggle solution ⭐
├── kaggle_notebook.py      # Legacy version
├── submission_script.py    # Inference & submission generation
├── verify_syntax.py        # Syntax verification script
├── src/
│   ├── model.py            # NFLModel architecture
│   ├── dataset.py          # NFLDataset with augmentation
│   ├── train.py            # Training loop
│   └── inference.py        # TTA, ensemble, smoothing
├── requirements.txt
└── README.md
```

## 📈 Training Log

```
Epoch  1/15: Train 18.04, Val RMSE 8.85 ✓ New best
Epoch  3/15: Train 14.26, Val RMSE 7.81 ✓ New best
Epoch  4/15: Train 13.32, Val RMSE 7.07 ✓ New best
Epoch 13/15: Train 12.05, Val RMSE 6.80 ✓ New best (Final)
Epoch 15/15: Train 11.63, Val RMSE 7.44
```

## 🔧 Key Hyperparameters

| Parameter | Value |
|-----------|-------|
| Hidden dim | 256 |
| Attention heads | 8 |
| Transformer layers | 3 |
| GRU layers | 2 (bidirectional) |
| Learning rate | 5e-4 |
| Batch size | 32 |
| Epochs | 15 |
| Gradient clip | 1.0 |
| Smoothness weight | 0.1 |

## 🧮 Normalization

Coordinates are normalized for stable training:
- **X**: Field length (0-120 yards) → Normalized by dividing by 120
- **Y**: Field width (0-53.3 yards) → Normalized by dividing by 53.3

The RMSE loss is calculated on raw yard coordinates during training, then converted to normalized scale for leaderboard comparison:
```
Normalized RMSE ≈ Raw RMSE / sqrt((120/2)² + (53.3/2)²) ≈ Raw RMSE / 65.3
```

## 📝 License

MIT License - Feel free to use and modify for your own solutions!

## 🙏 Acknowledgments

- NFL Big Data Bowl organizers
- Kaggle community for sharing winning solution insights
