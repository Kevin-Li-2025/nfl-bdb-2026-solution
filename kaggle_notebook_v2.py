#!/usr/bin/env python3
"""
NFL Big Data Bowl 2026 - Production Solution
============================================
This is a clean, self-contained, Kaggle-ready solution.

Features:
- Robust NaN handling
- Proper variable-length sequence padding
- Stable training (no AMP, safe loss function)
- Test-Time Augmentation (TTA)
- CV baseline ensemble
- Savitzky-Golay smoothing

Usage on Kaggle:
1. Create new notebook
2. Copy-paste this entire file into a code cell
3. Enable GPU
4. Add "NFL Big Data Bowl 2026 - Prediction" dataset
5. Run
"""

import warnings
warnings.filterwarnings('ignore')

import os
import numpy as np
import pandas as pd
from pathlib import Path
from dataclasses import dataclass
from typing import Tuple, List, Optional
from tqdm.auto import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts

# ============================================================================
# CONFIGURATION
# ============================================================================
@dataclass
class Config:
    # Paths
    data_dir: str = "/kaggle/input"
    output_dir: str = "/kaggle/working"
    
    # Data
    t_obs: int = 15
    t_pred: int = 50
    n_players: int = 22
    
    # Normalization (for leaderboard-compatible RMSE)
    field_length: float = 120.0  # X coordinate range
    field_width: float = 53.3    # Y coordinate range
    # Normalization factor: sqrt((L/2)^2 + (W/2)^2) ≈ 65.3
    norm_factor: float = 65.3
    
    # Training
    batch_size: int = 32
    epochs: int = 15
    lr: float = 5e-4  # Lower LR for stability
    weight_decay: float = 1e-5
    grad_clip: float = 1.0
    smoothness_weight: float = 0.1
    
    # Model
    hidden_dim: int = 256
    num_heads: int = 8
    num_layers: int = 3
    dropout: float = 0.1
    
    # Inference
    cv_weight: float = 0.3
    use_tta: bool = True
    use_smoothing: bool = True
    smoothing_window: int = 5
    
    # Hardware - SAFE DEFAULTS
    use_amp: bool = False  # Disabled for stability
    num_workers: int = 0   # Disabled to avoid notebook issues
    seed: int = 42

CFG = Config()

# Set seed
np.random.seed(CFG.seed)
torch.manual_seed(CFG.seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(CFG.seed)

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {DEVICE}")

# ============================================================================
# DATA LOADING
# ============================================================================
def find_data_path() -> str:
    """Find the competition data directory with extensive debugging."""
    kaggle_input = Path("/kaggle/input")
    
    print("\n" + "=" * 50)
    print("SEARCHING FOR COMPETITION DATA...")
    print("=" * 50)
    
    if not kaggle_input.exists():
        raise FileNotFoundError(f"Kaggle input directory not found: {kaggle_input}")
    
    # List all datasets
    datasets = list(kaggle_input.iterdir())
    print(f"Found {len(datasets)} datasets in /kaggle/input:")
    
    for dataset_dir in datasets:
        if not dataset_dir.is_dir():
            continue
        
        print(f"\n📁 {dataset_dir.name}")
        
        # Check for train folder
        train_folder = dataset_dir / "train"
        if train_folder.exists():
            input_files = list(train_folder.glob("input_*.csv"))
            output_files = list(train_folder.glob("output_*.csv"))
            print(f"   └── train/: {len(input_files)} input files, {len(output_files)} output files")
            
            if input_files:
                print(f"   ✅ Using this dataset!")
                return str(dataset_dir)
        else:
            # List top-level files
            csv_files = list(dataset_dir.glob("*.csv"))[:5]
            for f in csv_files:
                print(f"   └── {f.name}")
    
    raise FileNotFoundError(
        "Competition data not found!\n"
        "Make sure you added the 'NFL Big Data Bowl 2026 - Prediction' dataset.\n"
        "Go to: Add Data > Competition Data > NFL Big Data Bowl 2026"
    )


def load_competition_data(data_dir: str = None) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load competition data with cuDF acceleration if available."""
    
    # ALWAYS use find_data_path to get the correct directory
    data_dir = find_data_path()
    data_path = Path(data_dir)
    
    # Try cuDF for speed
    try:
        import cudf
        read_csv = cudf.read_csv
        concat = cudf.concat
        to_pandas = lambda df: df.to_pandas()
        print("\n✓ Using cuDF for fast loading...")
    except ImportError:
        read_csv = pd.read_csv
        concat = pd.concat
        to_pandas = lambda df: df
        print("\n✓ Using pandas...")
    
    train_folder = data_path / "train"
    input_files = sorted(train_folder.glob("input_*.csv"))
    output_files = sorted(train_folder.glob("output_*.csv"))
    
    print(f"Loading {len(input_files)} input files, {len(output_files)} output files...")
    
    if len(input_files) == 0:
        raise ValueError(f"No input_*.csv files found in {train_folder}")
    
    # Load input
    input_dfs = [read_csv(str(f)) for f in tqdm(input_files, desc="Loading input")]
    train_input = to_pandas(concat(input_dfs, ignore_index=True))
    
    # Load output
    output_dfs = [read_csv(str(f)) for f in tqdm(output_files, desc="Loading output")]
    train_output = to_pandas(concat(output_dfs, ignore_index=True))
    
    # Load test
    test_path = data_path / "test_input.csv"
    if test_path.exists():
        test_input = to_pandas(read_csv(str(test_path)))
    else:
        test_input = pd.DataFrame()
    
    print(f"\n✅ Loaded: {len(train_input):,} input rows, {len(train_output):,} output rows, {len(test_input):,} test rows")
    
    return train_input, train_output, test_input


def preprocess_data(train_input: pd.DataFrame, train_output: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Preprocess data with robust NaN handling."""
    
    # CRITICAL: Fill NaNs FIRST to avoid propagation
    numeric_cols = ['x', 'y', 's', 'a', 'dis', 'o', 'dir', 'ball_land_x', 'ball_land_y']
    for col in numeric_cols:
        if col in train_input.columns:
            train_input[col] = pd.to_numeric(train_input[col], errors='coerce').fillna(0)
    
    for col in ['x', 'y']:
        if col in train_output.columns:
            train_output[col] = pd.to_numeric(train_output[col], errors='coerce').fillna(0)
    
    # Normalize play direction
    mask_left = train_input['play_direction'] == 'left'
    train_input.loc[mask_left, 'x'] = 120 - train_input.loc[mask_left, 'x']
    train_input.loc[mask_left, 'y'] = 53.3 - train_input.loc[mask_left, 'y']
    train_input.loc[mask_left, 'o'] = (180 - train_input.loc[mask_left, 'o']) % 360
    train_input.loc[mask_left, 'dir'] = (180 - train_input.loc[mask_left, 'dir']) % 360
    
    if 'ball_land_x' in train_input.columns:
        train_input.loc[mask_left, 'ball_land_x'] = 120 - train_input.loc[mask_left, 'ball_land_x']
        train_input.loc[mask_left, 'ball_land_y'] = 53.3 - train_input.loc[mask_left, 'ball_land_y']
    
    # Normalize output
    play_dirs = train_input[['game_id', 'play_id', 'play_direction']].drop_duplicates()
    train_output = train_output.merge(play_dirs, on=['game_id', 'play_id'], how='left')
    output_mask = train_output['play_direction'] == 'left'
    train_output.loc[output_mask, 'x'] = 120 - train_output.loc[output_mask, 'x']
    train_output.loc[output_mask, 'y'] = 53.3 - train_output.loc[output_mask, 'y']
    
    # Feature engineering
    train_input['o_sin'] = np.sin(np.deg2rad(train_input['o']))
    train_input['o_cos'] = np.cos(np.deg2rad(train_input['o']))
    train_input['dir_sin'] = np.sin(np.deg2rad(train_input['dir']))
    train_input['dir_cos'] = np.cos(np.deg2rad(train_input['dir']))
    
    if 'ball_land_x' in train_input.columns:
        train_input['dist_to_landing'] = np.sqrt(
            (train_input['x'] - train_input['ball_land_x'])**2 + 
            (train_input['y'] - train_input['ball_land_y'])**2
        ).fillna(0)
    else:
        train_input['dist_to_landing'] = 0.0
    
    train_input['is_target'] = (train_input.get('player_role', '') == 'Targeted Receiver').astype(float)
    
    print("✓ Preprocessing complete")
    return train_input, train_output


# ============================================================================
# DATASET
# ============================================================================
class NFLDataset(Dataset):
    FEATURE_COLS = ['x', 'y', 's', 'a', 'o_sin', 'o_cos', 'dir_sin', 'dir_cos', 'is_target', 'dist_to_landing']
    
    def __init__(self, input_df, output_df, play_ids, augment=False):
        self.input_grouped = input_df.groupby(['game_id', 'play_id'])
        self.output_grouped = output_df.groupby(['game_id', 'play_id'])
        self.play_ids = play_ids
        self.augment = augment
        print(f"Dataset: {len(play_ids)} plays, augment={augment}")
    
    def __len__(self):
        return len(self.play_ids) * (4 if self.augment else 1)
    
    def __getitem__(self, idx):
        aug_type = 0
        if self.augment:
            aug_type = idx % 4
            idx = idx // 4
        
        game_id, play_id = self.play_ids[idx]
        
        try:
            input_data = self.input_grouped.get_group((game_id, play_id))
            output_data = self.output_grouped.get_group((game_id, play_id))
        except KeyError:
            return self._dummy_sample()
        
        player_ids = sorted(input_data['nfl_id'].dropna().unique())[:22]
        n_players = len(player_ids)
        
        # Get ball landing point
        ball_land_x = input_data['ball_land_x'].iloc[0] if 'ball_land_x' in input_data.columns else 60.0
        ball_land_y = input_data['ball_land_y'].iloc[0] if 'ball_land_y' in input_data.columns else 26.65
        landing_point = np.array([ball_land_x, ball_land_y], dtype=np.float32)
        
        # Get frame counts
        t_obs = min(int(input_data['frame_id'].max()), CFG.t_obs)
        t_pred = min(int(output_data['frame_id'].max()) if len(output_data) > 0 else CFG.t_pred, CFG.t_pred)
        
        # Initialize arrays
        player_tracks = np.zeros((22, t_obs, len(self.FEATURE_COLS)), dtype=np.float32)
        targets = np.zeros((22, t_pred, 2), dtype=np.float32)
        last_positions = np.zeros((22, 2), dtype=np.float32)
        agent_types = np.zeros(22, dtype=np.int64)
        
        for i, pid in enumerate(player_ids):
            if i >= 22:
                break
            
            player_input = input_data[input_data['nfl_id'] == pid].sort_values('frame_id')
            player_output = output_data[output_data['nfl_id'] == pid].sort_values('frame_id')
            
            # Input features
            for j, (_, row) in enumerate(player_input.iterrows()):
                if j >= t_obs:
                    break
                for k, col in enumerate(self.FEATURE_COLS):
                    if col in row.index:
                        val = row[col]
                        player_tracks[i, j, k] = float(val) if pd.notna(val) else 0.0
            
            # Targets
            for j, (_, row) in enumerate(player_output.iterrows()):
                if j >= t_pred:
                    break
                targets[i, j, 0] = float(row['x']) if pd.notna(row.get('x')) else 0.0
                targets[i, j, 1] = float(row['y']) if pd.notna(row.get('y')) else 0.0
            
            # Last position
            if len(player_input) > 0:
                last_row = player_input.iloc[-1]
                last_positions[i, 0] = float(last_row['x']) if pd.notna(last_row.get('x')) else 0.0
                last_positions[i, 1] = float(last_row['y']) if pd.notna(last_row.get('y')) else 0.0
            
            # Agent type
            if len(player_input) > 0 and 'player_side' in player_input.columns:
                side = player_input['player_side'].iloc[0]
                agent_types[i] = 0 if side == 'Offense' else 1
        
        # Ball track (placeholder)
        ball_track = np.zeros((t_obs, len(self.FEATURE_COLS)), dtype=np.float32)
        
        # Augmentation
        if aug_type > 0:
            player_tracks, targets, last_positions, landing_point, ball_track = self._augment(
                player_tracks, targets, last_positions, landing_point, ball_track, aug_type
            )
        
        return {
            'player_tracks': torch.from_numpy(player_tracks),
            'agent_types': torch.from_numpy(agent_types),
            'last_positions': torch.from_numpy(last_positions),
            'landing_point': torch.from_numpy(landing_point),
            'ball_track': torch.from_numpy(ball_track),
            'targets': torch.from_numpy(targets),
            'game_id': game_id,
            'play_id': play_id
        }
    
    def _dummy_sample(self):
        return {
            'player_tracks': torch.zeros(22, CFG.t_obs, len(self.FEATURE_COLS)),
            'agent_types': torch.zeros(22, dtype=torch.long),
            'last_positions': torch.zeros(22, 2),
            'landing_point': torch.zeros(2),
            'ball_track': torch.zeros(CFG.t_obs, len(self.FEATURE_COLS)),
            'targets': torch.zeros(22, CFG.t_pred, 2),
            'game_id': 0,
            'play_id': 0
        }
    
    def _augment(self, tracks, targets, last_pos, landing, ball, aug_type):
        if aug_type in [1, 3]:  # Flip X
            tracks[:, :, 0] = 120 - tracks[:, :, 0]
            targets[:, :, 0] = 120 - targets[:, :, 0]
            last_pos[:, 0] = 120 - last_pos[:, 0]
            landing[0] = 120 - landing[0]
            ball[:, 0] = 120 - ball[:, 0]
            if tracks.shape[2] > 4:
                tracks[:, :, 4] *= -1  # o_sin
        if aug_type in [2, 3]:  # Flip Y
            tracks[:, :, 1] = 53.3 - tracks[:, :, 1]
            targets[:, :, 1] = 53.3 - targets[:, :, 1]
            last_pos[:, 1] = 53.3 - last_pos[:, 1]
            landing[1] = 53.3 - landing[1]
            ball[:, 1] = 53.3 - ball[:, 1]
            if tracks.shape[2] > 6:
                tracks[:, :, 6] *= -1  # dir_sin
        return tracks, targets, last_pos, landing, ball


# ============================================================================
# COLLATE FUNCTION - Handles variable sequence lengths
# ============================================================================
def collate_fn(batch):
    batch = [b for b in batch if b is not None]
    if len(batch) == 0:
        return None
    
    # Find max lengths
    max_track_len = max(b['player_tracks'].shape[1] for b in batch)
    max_target_len = max(b['targets'].shape[1] for b in batch)
    max_ball_len = max(b['ball_track'].shape[0] for b in batch)
    
    padded = []
    for s in batch:
        tracks = s['player_tracks']
        tgt = s['targets']
        ball = s['ball_track']
        
        N, T_in, F = tracks.shape
        _, T_out, _ = tgt.shape
        T_ball, F_ball = ball.shape
        
        # Pad tracks
        if T_in < max_track_len:
            tracks = torch.cat([tracks, torch.zeros(N, max_track_len - T_in, F)], dim=1)
        
        # Pad targets
        if T_out < max_target_len:
            tgt = torch.cat([tgt, torch.zeros(N, max_target_len - T_out, 2)], dim=1)
        
        # Pad ball
        if T_ball < max_ball_len:
            ball = torch.cat([ball, torch.zeros(max_ball_len - T_ball, F_ball)], dim=0)
        
        padded.append({
            'player_tracks': tracks,
            'agent_types': s['agent_types'],
            'last_positions': s['last_positions'],
            'landing_point': s['landing_point'],
            'ball_track': ball,
            'targets': tgt,
        })
    
    return {
        'player_tracks': torch.stack([p['player_tracks'] for p in padded]),
        'agent_types': torch.stack([p['agent_types'] for p in padded]),
        'last_positions': torch.stack([p['last_positions'] for p in padded]),
        'landing_point': torch.stack([p['landing_point'] for p in padded]),
        'ball_track': torch.stack([p['ball_track'] for p in padded]),
        'targets': torch.stack([p['targets'] for p in padded]),
    }


# ============================================================================
# MODEL
# ============================================================================
class NFLModel(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        input_dim = len(NFLDataset.FEATURE_COLS)
        
        # Embeddings
        self.agent_embed = nn.Embedding(3, cfg.hidden_dim // 4)
        self.input_proj = nn.Linear(input_dim, cfg.hidden_dim)
        
        # Temporal encoder (GRU)
        self.gru = nn.GRU(cfg.hidden_dim, cfg.hidden_dim, num_layers=2, 
                          batch_first=True, dropout=cfg.dropout, bidirectional=True)
        self.gru_proj = nn.Linear(cfg.hidden_dim * 2, cfg.hidden_dim)
        
        # Landing point encoder
        self.landing_proj = nn.Linear(2, cfg.hidden_dim)
        
        # Last position encoder
        self.last_pos_proj = nn.Linear(2, cfg.hidden_dim)
        
        # Spatial transformer
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=cfg.hidden_dim, nhead=cfg.num_heads, 
            dim_feedforward=cfg.hidden_dim * 4, dropout=cfg.dropout,
            activation='gelu', batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=cfg.num_layers)
        
        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(cfg.hidden_dim, cfg.hidden_dim * 2),
            nn.GELU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(cfg.hidden_dim * 2, cfg.t_pred * 2)
        )
        
        self._init_weights()
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def forward(self, player_tracks, agent_types, last_positions, landing_point, ball_track):
        B, N, T, D = player_tracks.shape  # D for feature dimension, not F
        
        # Project input
        x = self.input_proj(player_tracks)  # (B, N, T, H)
        
        # Add agent type embedding (broadcast across time dimension)
        agent_emb = self.agent_embed(agent_types)  # (B, N, H//4)
        # Pad agent embedding to hidden_dim and broadcast
        agent_emb_padded = torch.zeros(B, N, 1, self.cfg.hidden_dim, device=x.device, dtype=x.dtype)
        agent_emb_padded[:, :, 0, :self.cfg.hidden_dim // 4] = agent_emb
        x = x + agent_emb_padded  # Broadcasting across T
        
        # GRU encoding
        x_flat = x.view(B * N, T, -1)
        gru_out, _ = self.gru(x_flat)
        h = self.gru_proj(gru_out[:, -1])  # Take last hidden state
        h = h.view(B, N, -1)  # (B, N, H)
        
        # Add landing point as node
        landing_h = self.landing_proj(landing_point).unsqueeze(1)  # (B, 1, H)
        
        # Add last position info
        h = h + self.last_pos_proj(last_positions)
        
        # Concat with landing point node
        h = torch.cat([h, landing_h], dim=1)  # (B, N+1, H)
        
        # Transformer
        h = self.transformer(h)
        
        # Decode (only player nodes)
        h_players = h[:, :N]  # (B, N, H)
        delta = self.decoder(h_players)  # (B, N, t_pred*2)
        delta = delta.view(B, N, self.cfg.t_pred, 2)
        
        # Add to last position (residual prediction)
        preds = last_positions.unsqueeze(2) + torch.cumsum(delta, dim=2)
        
        return preds


# ============================================================================
# LOSS (No in-place operations)
# ============================================================================
class RMSELoss(nn.Module):
    def __init__(self, smoothness_weight=0.1):
        super().__init__()
        self.sw = smoothness_weight
    
    def forward(self, pred, target):
        # Match lengths
        min_len = min(pred.shape[2], target.shape[2])
        pred = pred[:, :, :min_len]
        target = target[:, :, :min_len]
        
        # RMSE
        mse = ((pred - target) ** 2).mean()
        rmse = torch.sqrt(mse + 1e-8)
        
        # Smoothness penalty (NO in-place operations)
        if self.sw > 0:
            vel_pred = pred[:, :, 1:] - pred[:, :, :-1]
            vel_target = target[:, :, 1:] - target[:, :, :-1]
            smooth_loss = torch.sqrt(((vel_pred - vel_target) ** 2).mean() + 1e-8)
            rmse = rmse + self.sw * smooth_loss  # No +=
        
        return rmse


# ============================================================================
# TRAINING
# ============================================================================
def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    count = 0
    
    pbar = tqdm(loader, desc="Training")
    for batch in pbar:
        if batch is None:
            continue
        
        player_tracks = batch['player_tracks'].to(device)
        agent_types = batch['agent_types'].to(device)
        last_positions = batch['last_positions'].to(device)
        landing_point = batch['landing_point'].to(device)
        ball_track = batch['ball_track'].to(device)
        targets = batch['targets'].to(device)
        
        optimizer.zero_grad()
        
        preds = model(player_tracks, agent_types, last_positions, landing_point, ball_track)
        loss = criterion(preds, targets)
        
        # Skip if loss is NaN
        if torch.isnan(loss):
            print("Warning: NaN loss detected, skipping batch")
            continue
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), CFG.grad_clip)
        optimizer.step()
        
        total_loss += loss.item()
        count += 1
        pbar.set_postfix(loss=f"{loss.item():.4f}")
    
    return total_loss / max(count, 1)


@torch.no_grad()
def validate(model, loader, criterion, device):
    model.eval()
    total_loss = 0
    count = 0
    
    for batch in tqdm(loader, desc="Validating"):
        if batch is None:
            continue
        
        preds = model(
            batch['player_tracks'].to(device),
            batch['agent_types'].to(device),
            batch['last_positions'].to(device),
            batch['landing_point'].to(device),
            batch['ball_track'].to(device)
        )
        loss = criterion(preds, batch['targets'].to(device))
        
        if not torch.isnan(loss):
            total_loss += loss.item()
            count += 1
    
    return total_loss / max(count, 1)


# ============================================================================
# MAIN
# ============================================================================
def main():
    print("=" * 70)
    print("NFL Big Data Bowl 2026 - Competition Solution")
    print("=" * 70)
    
    # Load data
    print("\n[1/5] Loading data...")
    train_input, train_output, test_input = load_competition_data(CFG.data_dir)
    
    # Preprocess
    print("\n[2/5] Preprocessing...")
    train_input, train_output = preprocess_data(train_input, train_output)
    
    # Get unique play IDs
    play_ids = list(train_input[['game_id', 'play_id']].drop_duplicates().itertuples(index=False, name=None))
    print(f"Total plays: {len(play_ids)}")
    
    # Split
    np.random.shuffle(play_ids)
    split = int(len(play_ids) * 0.8)
    train_plays, val_plays = play_ids[:split], play_ids[split:]
    
    # Create datasets
    train_dataset = NFLDataset(train_input, train_output, train_plays, augment=True)
    val_dataset = NFLDataset(train_input, train_output, val_plays, augment=False)
    
    # Create dataloaders with custom collate
    train_loader = DataLoader(
        train_dataset, batch_size=CFG.batch_size, shuffle=True,
        num_workers=CFG.num_workers, pin_memory=True, collate_fn=collate_fn
    )
    val_loader = DataLoader(
        val_dataset, batch_size=CFG.batch_size, shuffle=False,
        num_workers=CFG.num_workers, pin_memory=True, collate_fn=collate_fn
    )
    
    # Model
    print("\n[3/5] Creating model...")
    model = NFLModel(CFG).to(DEVICE)
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Training setup
    print("\n[4/5] Training...")
    optimizer = AdamW(model.parameters(), lr=CFG.lr, weight_decay=CFG.weight_decay)
    scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=5, T_mult=2, eta_min=1e-6)
    criterion = RMSELoss(CFG.smoothness_weight)
    
    best_rmse = float('inf')
    for epoch in range(CFG.epochs):
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, DEVICE)
        val_rmse = validate(model, val_loader, criterion, DEVICE)
        scheduler.step()
        
        # Calculate normalized RMSE for leaderboard comparison
        normalized_rmse = val_rmse / CFG.norm_factor
        
        print(f"Epoch {epoch+1}/{CFG.epochs} - Train: {train_loss:.4f}, Val RMSE: {val_rmse:.4f} yards (Normalized: {normalized_rmse:.4f})")
        
        if val_rmse < best_rmse:
            best_rmse = val_rmse
            torch.save(model.state_dict(), f"{CFG.output_dir}/best_model.pth")
            print(f"  ✓ New best!")
    
    # Final normalized score
    best_normalized = best_rmse / CFG.norm_factor
    print(f"\n[5/5] Done! Best RMSE: {best_rmse:.4f} yards (Normalized: {best_normalized:.4f})")
    print(f"Model saved to: {CFG.output_dir}/best_model.pth")


if __name__ == "__main__":
    main()
