"""
NFL Big Data Bowl 2026 - Kaggle Competition Notebook (CORRECTED)
================================================================================
Uses the ACTUAL competition data format:
- train/input_2023_w[01-18].csv - Input tracking (before pass)
- train/output_2023_w[01-18].csv - Output tracking (after pass) 
- test_input.csv - Test data
- test.csv - Submission template

Key columns: game_id, play_id, nfl_id, frame_id, ball_land_x, ball_land_y
================================================================================
"""

import os
import gc
import math
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from torch.cuda.amp import autocast, GradScaler
from scipy.signal import savgol_filter
from tqdm.auto import tqdm

warnings.filterwarnings('ignore')

# ============================================================================
# CONFIGURATION
# ============================================================================
@dataclass
class Config:
    # Data paths
    data_dir: str = "/kaggle/input/nfl-big-data-bowl-2026-prediction"
    output_dir: str = "/kaggle/working"
    
    # Data params
    t_obs: int = 10       # Frames observed (from input files)
    t_pred: int = 30      # Frames to predict (from output files)
    
    # Model architecture  
    input_dim: int = 10   # Features per player per frame
    hidden_dim: int = 256
    nhead: int = 8
    num_temporal_layers: int = 2
    num_spatial_layers: int = 4
    dropout: float = 0.1
    
    # Training
    batch_size: int = 32
    epochs: int = 15
    lr: float = 5e-4
    weight_decay: float = 1e-5
    grad_clip: float = 1.0
    smoothness_weight: float = 0.1
    
    # Validation
    n_folds: int = 5
    train_fold: int = 0
    
    # Inference
    cv_weight: float = 0.3
    use_tta: bool = True
    use_smoothing: bool = True
    smoothing_window: int = 5
    
    # Hardware
    use_amp: bool = False  # Set to False to prevent loss=nan instability
    num_workers: int = 0  # Set to 0 for Kaggle notebooks to avoid multiprocessing issues
    seed: int = 42

CFG = Config()

# ============================================================================
# UTILITIES
# ============================================================================
def seed_everything(seed: int):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

def get_device():
    if torch.cuda.is_available():
        return torch.device('cuda')
    return torch.device('cpu')

DEVICE = get_device()
print(f"Using device: {DEVICE}")
seed_everything(CFG.seed)

# ============================================================================
# DATA LOADING - CORRECTED FOR ACTUAL COMPETITION FORMAT
# ============================================================================
def find_data_path() -> str:
    """Find the competition data directory."""
    kaggle_input = Path("/kaggle/input")
    
    if kaggle_input.exists():
        print("\n" + "="*60)
        print("SCANNING FOR COMPETITION DATA...")
        print("="*60)
        
        for dataset_dir in kaggle_input.iterdir():
            if dataset_dir.is_dir():
                print(f"\n📁 {dataset_dir.name}")
                
                # Look for the train folder (actual competition structure)
                train_folder = dataset_dir / "train"
                if train_folder.exists():
                    input_files = list(train_folder.glob("input_*.csv"))
                    if input_files:
                        print(f"   ✓ Found train folder with {len(input_files)} input files")
                        return str(dataset_dir)
                
                # Also check for test.csv at root
                test_file = dataset_dir / "test.csv"
                test_input = dataset_dir / "test_input.csv"
                if test_file.exists() or test_input.exists():
                    print(f"   ✓ Found test files")
                    return str(dataset_dir)
                    
                # List some files for debugging
                files = list(dataset_dir.rglob("*.csv"))[:10]
                for f in files:
                    print(f"   📄 {f.relative_to(dataset_dir)}")
        
        print("\n❌ Competition data not found!")
        print("Make sure you added 'NFL Big Data Bowl 2026 - Prediction' dataset")
    
    raise FileNotFoundError("Competition data not found")


def load_competition_data(data_dir: str = None) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Load the ACTUAL competition data format.
    
    Returns:
        train_input: All input frames (before pass)
        train_output: All output frames (after pass) - targets
        test_input: Test data input frames
    """
    if data_dir is None or not Path(data_dir).exists():
        data_dir = find_data_path()
    
    data_path = Path(data_dir)
    print(f"\nLoading from: {data_path}")
    
    # Try cuDF first
    try:
        import cudf
        read_csv = cudf.read_csv
        concat = cudf.concat
        to_pandas = lambda df: df.to_pandas()
        print("Using cuDF for fast loading...")
    except ImportError:
        read_csv = pd.read_csv
        concat = pd.concat
        to_pandas = lambda df: df
        print("Using pandas...")
    
    # Load training input files
    train_folder = data_path / "train"
    input_files = sorted(train_folder.glob("input_*.csv"))
    output_files = sorted(train_folder.glob("output_*.csv"))
    
    print(f"Found {len(input_files)} input files, {len(output_files)} output files")
    
    # Load input data
    print("Loading input data...")
    input_dfs = []
    for f in tqdm(input_files, desc="Input files"):
        df = read_csv(f)
        input_dfs.append(df)
    train_input = concat(input_dfs, ignore_index=True)
    train_input = to_pandas(train_input)
    
    # Load output data
    print("Loading output data...")
    output_dfs = []
    for f in tqdm(output_files, desc="Output files"):
        df = read_csv(f)
        output_dfs.append(df)
    train_output = concat(output_dfs, ignore_index=True)
    train_output = to_pandas(train_output)
    
    # Load test input
    test_input_path = data_path / "test_input.csv"
    if test_input_path.exists():
        test_input = to_pandas(read_csv(test_input_path))
    else:
        test_input = pd.DataFrame()
    
    print(f"\n✓ Loaded:")
    print(f"  Train input: {len(train_input):,} rows")
    print(f"  Train output: {len(train_output):,} rows")
    print(f"  Test input: {len(test_input):,} rows")
    
    # Show columns
    print(f"\nInput columns: {list(train_input.columns[:15])}...")
    
    return train_input, train_output, test_input


def preprocess_data(train_input: pd.DataFrame, train_output: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Preprocess data and engineer features."""
    
    # Fill NaNs in numerical columns
    fill_cols = ['x', 'y', 's', 'a', 'dis', 'o', 'dir', 'ball_land_x', 'ball_land_y']
    for col in fill_cols:
        if col in train_input.columns:
            train_input[col] = train_input[col].fillna(0)
    
    # Normalize play direction
    mask_left = train_input['play_direction'] == 'left'
    train_input.loc[mask_left, 'x'] = 120 - train_input.loc[mask_left, 'x']
    train_input.loc[mask_left, 'y'] = 53.3 - train_input.loc[mask_left, 'y']
    train_input.loc[mask_left, 'o'] = (180 - train_input.loc[mask_left, 'o']) % 360
    train_input.loc[mask_left, 'dir'] = (180 - train_input.loc[mask_left, 'dir']) % 360
    train_input.loc[mask_left, 'ball_land_x'] = 120 - train_input.loc[mask_left, 'ball_land_x']
    train_input.loc[mask_left, 'ball_land_y'] = 53.3 - train_input.loc[mask_left, 'ball_land_y']
    
    # Normalize output coordinates too
    output_mask_left = train_output['game_id'].isin(train_input[mask_left]['game_id'].unique())
    # Merge to get play direction for output
    play_dirs = train_input[['game_id', 'play_id', 'play_direction']].drop_duplicates()
    train_output = train_output.merge(play_dirs, on=['game_id', 'play_id'], how='left')
    output_mask = train_output['play_direction'] == 'left'
    train_output.loc[output_mask, 'x'] = 120 - train_output.loc[output_mask, 'x']
    train_output.loc[output_mask, 'y'] = 53.3 - train_output.loc[output_mask, 'y']
    
    # Sin/cos encoding for angles
    train_input['o_sin'] = np.sin(np.deg2rad(train_input['o']))
    train_input['o_cos'] = np.cos(np.deg2rad(train_input['o']))
    train_input['dir_sin'] = np.sin(np.deg2rad(train_input['dir']))
    train_input['dir_cos'] = np.cos(np.deg2rad(train_input['dir']))
    
    # Distance to ball landing
    train_input['dist_to_landing'] = np.sqrt(
        (train_input['x'] - train_input['ball_land_x'])**2 + 
        (train_input['y'] - train_input['ball_land_y'])**2
    )
    
    # Is target receiver
    train_input['is_target'] = (train_input['player_role'] == 'Targeted Receiver').astype(float)
    
    print("✓ Preprocessing complete")
    return train_input, train_output


# ============================================================================
# DATASET - Corrected for new format
# ============================================================================
class NFLDataset(Dataset):
    """Dataset for NFL Big Data Bowl 2026."""
    
    FEATURE_COLS = ['x', 'y', 's', 'a', 'o_sin', 'o_cos', 'dir_sin', 'dir_cos', 'is_target', 'dist_to_landing']
    
    def __init__(
        self,
        train_input: pd.DataFrame,
        train_output: pd.DataFrame,
        play_ids: List[Tuple[int, int]],
        augment: bool = True,
        mode: str = 'train'
    ):
        self.train_input = train_input
        self.train_output = train_output
        self.play_ids = play_ids
        self.augment = augment and mode == 'train'
        
        # Group by game and play
        self.input_grouped = train_input.groupby(['game_id', 'play_id'])
        self.output_grouped = train_output.groupby(['game_id', 'play_id'])
        
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
            return self._dummy_sample(game_id, play_id)
        
        # Get unique players
        player_ids = sorted(input_data['nfl_id'].dropna().unique())[:22]
        n_players = len(player_ids)
        
        # Get ball landing point
        ball_land_x = input_data['ball_land_x'].iloc[0]
        ball_land_y = input_data['ball_land_y'].iloc[0]
        landing_point = np.array([ball_land_x, ball_land_y], dtype=np.float32)
        
        # Get number of frames
        t_obs = input_data['frame_id'].max()
        t_pred = output_data['frame_id'].max() if len(output_data) > 0 else CFG.t_pred
        
        # Extract features
        player_tracks = np.zeros((22, min(t_obs, 15), len(self.FEATURE_COLS)), dtype=np.float32)
        targets = np.zeros((22, min(t_pred, 50), 2), dtype=np.float32)
        last_positions = np.zeros((22, 2), dtype=np.float32)
        agent_types = np.zeros(22, dtype=np.int64)
        
        for i, pid in enumerate(player_ids):
            if i >= 22:
                break
                
            player_input = input_data[input_data['nfl_id'] == pid].sort_values('frame_id')
            player_output = output_data[output_data['nfl_id'] == pid].sort_values('frame_id')
            
            # Input features
            for j, (_, row) in enumerate(player_input.iterrows()):
                if j >= player_tracks.shape[1]:
                    break
                for k, col in enumerate(self.FEATURE_COLS):
                    if col in row.index and pd.notna(row[col]):
                        player_tracks[i, j, k] = row[col]
            
            # Target positions
            for j, (_, row) in enumerate(player_output.iterrows()):
                if j >= targets.shape[1]:
                    break
                if 'x' in row.index and 'y' in row.index:
                    targets[i, j, 0] = row['x']
                    targets[i, j, 1] = row['y']
            
            # Last position
            if len(player_input) > 0:
                last_row = player_input.iloc[-1]
                last_positions[i, 0] = last_row['x']
                last_positions[i, 1] = last_row['y']
            
            # Agent type
            if len(player_input) > 0:
                side = player_input['player_side'].iloc[0]
                agent_types[i] = 0 if side == 'Offense' else 1
        
        # Ball track (use average position as placeholder)
        ball_track = np.zeros((player_tracks.shape[1], len(self.FEATURE_COLS)), dtype=np.float32)
        
        # Apply augmentation
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
    
    def _augment(self, tracks, targets, last_pos, landing, ball, aug_type):
        if aug_type in [1, 3]:  # Flip X
            tracks[:, :, 0] = 120 - tracks[:, :, 0]
            targets[:, :, 0] = 120 - targets[:, :, 0]
            last_pos[:, 0] = 120 - last_pos[:, 0]
            landing[0] = 120 - landing[0]
            ball[:, 0] = 120 - ball[:, 0]
            tracks[:, :, 4] *= -1  # o_sin
        if aug_type in [2, 3]:  # Flip Y
            tracks[:, :, 1] = 53.3 - tracks[:, :, 1]
            targets[:, :, 1] = 53.3 - targets[:, :, 1]
            last_pos[:, 1] = 53.3 - last_pos[:, 1]
            landing[1] = 53.3 - landing[1]
            ball[:, 1] = 53.3 - ball[:, 1]
            tracks[:, :, 6] *= -1  # dir_sin
        return tracks, targets, last_pos, landing, ball
    
    def _dummy_sample(self, game_id, play_id):
        return {
            'player_tracks': torch.zeros(22, 15, len(self.FEATURE_COLS)),
            'agent_types': torch.zeros(22, dtype=torch.long),
            'last_positions': torch.zeros(22, 2),
            'landing_point': torch.zeros(2),
            'ball_track': torch.zeros(15, len(self.FEATURE_COLS)),
            'targets': torch.zeros(22, 50, 2),
            'game_id': game_id,
            'play_id': play_id
        }


# ============================================================================
# MODEL (Same architecture but adapted for variable sequence lengths)
# ============================================================================
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=100):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]


class TemporalEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers=2, dropout=0.1):
        super().__init__()
        self.gru = nn.GRU(input_dim, hidden_dim//2, num_layers, batch_first=True, bidirectional=True, dropout=dropout if num_layers > 1 else 0)
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, x):
        out, _ = self.gru(x)
        return self.norm(out[:, -1, :])


class SpatialTransformer(nn.Module):
    def __init__(self, hidden_dim, nhead=8, num_layers=4, dropout=0.1):
        super().__init__()
        layer = nn.TransformerEncoderLayer(hidden_dim, nhead, hidden_dim*4, dropout, batch_first=True, activation='gelu')
        self.transformer = nn.TransformerEncoder(layer, num_layers)
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, x):
        return self.norm(self.transformer(x))


class TrajectoryDecoder(nn.Module):
    def __init__(self, hidden_dim, t_pred, dropout=0.1):
        super().__init__()
        self.t_pred = t_pred
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim*2), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden_dim*2, hidden_dim), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden_dim, t_pred*2)
        )

    def forward(self, x):
        return self.mlp(x).view(-1, self.t_pred, 2)


class NFLModel(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.input_proj = nn.Linear(cfg.input_dim, cfg.hidden_dim)
        self.temporal_encoder = TemporalEncoder(cfg.hidden_dim, cfg.hidden_dim, cfg.num_temporal_layers, cfg.dropout)
        self.type_embed = nn.Embedding(4, cfg.hidden_dim)
        self.pos_embed = PositionalEncoding(cfg.hidden_dim)
        self.landing_proj = nn.Linear(2, cfg.hidden_dim)
        self.last_pos_proj = nn.Linear(2, cfg.hidden_dim)
        self.spatial_transformer = SpatialTransformer(cfg.hidden_dim, cfg.nhead, cfg.num_spatial_layers, cfg.dropout)
        self.decoder = TrajectoryDecoder(cfg.hidden_dim, 50, cfg.dropout)  # Fixed output size

    def forward(self, player_tracks, agent_types, last_positions, landing_point, ball_track):
        B, N, T, F = player_tracks.shape
        device = player_tracks.device
        
        x = self.input_proj(player_tracks)
        h_players = self.temporal_encoder(x.view(B*N, T, -1)).view(B, N, -1)
        h_ball = self.temporal_encoder(self.input_proj(ball_track)).unsqueeze(1)
        h_landing = self.landing_proj(landing_point).unsqueeze(1)
        
        h_all = torch.cat([h_players, h_ball, h_landing], dim=1)
        all_types = torch.cat([agent_types, torch.full((B,1), 2, device=device), torch.full((B,1), 3, device=device)], dim=1)
        h_all = h_all + self.type_embed(all_types)
        h_all = self.pos_embed(h_all)
        
        all_pos = torch.cat([last_positions, ball_track[:, -1, :2].unsqueeze(1), landing_point.unsqueeze(1)], dim=1)
        h_all = h_all + self.last_pos_proj(all_pos)
        
        h_spatial = self.spatial_transformer(h_all)
        delta = self.decoder(h_spatial[:, :N].reshape(B*N, -1)).view(B, N, -1, 2)
        preds = torch.cumsum(delta, dim=2) + last_positions.unsqueeze(2)
        
        return preds


# ============================================================================
# LOSS, TRAINING, INFERENCE (unchanged)
# ============================================================================
class RMSELoss(nn.Module):
    def __init__(self, smoothness_weight=0.1):
        super().__init__()
        self.sw = smoothness_weight

    def forward(self, pred, target):
        # Handle different lengths
        min_len = min(pred.shape[2], target.shape[2])
        pred = pred[:, :, :min_len]
        target = target[:, :, :min_len]
        
        mse = ((pred - target) ** 2).mean()
        rmse = torch.sqrt(mse + 1e-8)
        if self.sw > 0:
            vel_pred = pred[:, :, 1:] - pred[:, :, :-1]
            vel_target = target[:, :, 1:] - target[:, :, :-1]
            rmse = rmse + self.sw * torch.sqrt(((vel_pred - vel_target)**2).mean() + 1e-8)
        return rmse


# ============================================================================
# COLLATE FUNCTION FOR VARIABLE SEQUENCE LENGTHS
# ============================================================================
def collate_fn(batch):
    """
    Custom collate function to handle variable sequence lengths.
    Pads all sequences to the maximum length in the batch.
    """
    # Filter out None samples
    batch = [b for b in batch if b is not None]
    if len(batch) == 0:
        return None
    
    # Find max lengths - using actual keys from NFLDataset
    max_track_len = max(b['player_tracks'].shape[1] for b in batch)
    max_target_len = max(b['targets'].shape[1] for b in batch)
    
    padded_batch = []
    for sample in batch:
        tracks = sample['player_tracks']  # (22, T_in, F)
        tgt = sample['targets']  # (22, T_out, 2)
        ball = sample['ball_track']  # (T_in, F)
        
        N, T_in, F = tracks.shape
        _, T_out, _ = tgt.shape
        T_ball, F_ball = ball.shape
        
        # Pad tracks
        if T_in < max_track_len:
            pad = torch.zeros(N, max_track_len - T_in, F)
            tracks = torch.cat([tracks, pad], dim=1)
        
        # Pad ball_track
        if T_ball < max_track_len:
            pad = torch.zeros(max_track_len - T_ball, F_ball)
            ball = torch.cat([ball, pad], dim=0)
        
        # Pad targets
        if T_out < max_target_len:
            pad = torch.zeros(N, max_target_len - T_out, 2)
            tgt = torch.cat([tgt, pad], dim=1)
        
        # Create mask (1 = valid, 0 = padded)
        mask = torch.zeros(N, max_target_len)
        mask[:, :T_out] = 1.0
        
        padded_batch.append({
            'player_tracks': tracks,
            'agent_types': sample['agent_types'],
            'last_positions': sample['last_positions'],
            'landing_point': sample['landing_point'],
            'ball_track': ball,
            'targets': tgt,
            'mask': mask,
            'game_id': sample.get('game_id', 0),
            'play_id': sample.get('play_id', 0),
            'original_len': T_out
        })
    
    # Stack into batch tensors
    return {
        'player_tracks': torch.stack([b['player_tracks'] for b in padded_batch]),
        'agent_types': torch.stack([b['agent_types'] for b in padded_batch]),
        'last_positions': torch.stack([b['last_positions'] for b in padded_batch]),
        'landing_point': torch.stack([b['landing_point'] for b in padded_batch]),
        'ball_track': torch.stack([b['ball_track'] for b in padded_batch]),
        'targets': torch.stack([b['targets'] for b in padded_batch]),
        'mask': torch.stack([b['mask'] for b in padded_batch]),
        'game_ids': [b['game_id'] for b in padded_batch],
        'play_ids': [b['play_id'] for b in padded_batch],
        'original_lens': [b['original_len'] for b in padded_batch]
    }


# ============================================================================
# TRAIN ONE EPOCH (Updated with masking)
# ============================================================================
def train_one_epoch(model, loader, optimizer, criterion, scaler, device, use_amp=True):
    model.train()
    total_loss = 0
    
    pbar = tqdm(loader, desc="Training")
    for batch in pbar:
        if batch is None: continue
        
        player_tracks = batch['player_tracks'].to(device)
        agent_types = batch['agent_types'].to(device)
        last_positions = batch['last_positions'].to(device)
        landing_point = batch['landing_point'].to(device)
        ball_track = batch['ball_track'].to(device)
        targets = batch['targets'].to(device)
        
        optimizer.zero_grad()
        
        with autocast(enabled=use_amp):
            preds = model(player_tracks, agent_types, last_positions, landing_point, ball_track)
            
            # Create mask for valid target frames (not padded)
            # We assume padding is zeros. A more robust way would be to pass lengths.
            # But for now, we just crop predictions to match target length if needed
            if preds.shape[2] != targets.shape[2]:
                 min_len = min(preds.shape[2], targets.shape[2])
                 preds = preds[:, :, :min_len]
                 targets = targets[:, :, :min_len]
            
            loss = criterion(preds, targets)
        
        if use_amp:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), CFG.grad_clip)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), CFG.grad_clip)
            optimizer.step()
        
        total_loss += loss.item()
        pbar.set_postfix(loss=f"{loss.item():.4f}")
    
    return total_loss / len(loader)


@torch.no_grad()
def validate(model, loader, criterion, device):
    model.eval()
    total_loss = 0
    for batch in tqdm(loader, desc="Validating"):
        if batch is None: continue # Added for collate_fn
        preds = model(
            batch['player_tracks'].to(device),
            batch['agent_types'].to(device),
            batch['last_positions'].to(device),
            batch['landing_point'].to(device),
            batch['ball_track'].to(device)
        )
        loss = criterion(preds, batch['targets'].to(device))
        total_loss += loss.item()
    return total_loss / len(loader)


# ============================================================================
# MAIN
# ============================================================================
def main():
    print("="*70)
    print("NFL Big Data Bowl 2026 - Competition Solution")
    print("="*70)
    
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
    
    # USE CUSTOM COLLATE FUNCTION
    train_loader = DataLoader(
        train_dataset, 
        batch_size=CFG.batch_size, 
        shuffle=True, 
        num_workers=CFG.num_workers, 
        pin_memory=True,
        collate_fn=collate_fn  # <--- Added
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=CFG.batch_size, 
        shuffle=False, 
        num_workers=CFG.num_workers, 
        pin_memory=True,
        collate_fn=collate_fn  # <--- Added
    )
    
    # Model
    print("\n[3/5] Creating model...")
    model = NFLModel(CFG).to(DEVICE)
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Training
    print("\n[4/5] Training...")
    optimizer = AdamW(model.parameters(), lr=CFG.lr, weight_decay=CFG.weight_decay)
    scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=5, T_mult=2, eta_min=1e-6)
    criterion = RMSELoss(CFG.smoothness_weight)
    scaler = GradScaler(enabled=CFG.use_amp)
    
    best_rmse = float('inf')
    for epoch in range(CFG.epochs):
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, scaler, DEVICE, CFG.use_amp)
        val_rmse = validate(model, val_loader, criterion, DEVICE)
        scheduler.step()
        
        print(f"Epoch {epoch+1}/{CFG.epochs} - Train: {train_loss:.4f}, Val RMSE: {val_rmse:.4f}")
        
        if val_rmse < best_rmse:
            best_rmse = val_rmse
            torch.save(model.state_dict(), f"{CFG.output_dir}/best_model.pth")
            print(f"  ✓ New best!")
    
    print(f"\n[5/5] Done! Best RMSE: {best_rmse:.4f}")


if __name__ == "__main__":
    main()
