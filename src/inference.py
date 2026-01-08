"""
NFL Big Data Bowl 2026 - Inference with TTA, CV Ensemble, and Smoothing
Based on winning solution post-processing techniques.
"""
import torch
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.signal import savgol_filter
from typing import Dict, List, Tuple, Optional
from tqdm import tqdm

from model import OptimizedSTGraphTransformer


class ConstantVelocityBaseline:
    """
    Constant Velocity (CV) baseline predictor.
    Simply extrapolates the last observed velocity.
    """
    @staticmethod
    def predict(last_positions: np.ndarray, last_velocities: np.ndarray, t_pred: int) -> np.ndarray:
        """
        Args:
            last_positions: [N, 2] last known (x, y)
            last_velocities: [N, 2] last velocity (dx, dy per frame)
            t_pred: number of frames to predict
        
        Returns:
            predictions: [N, t_pred, 2] predicted coordinates
        """
        N = last_positions.shape[0]
        predictions = np.zeros((N, t_pred, 2))
        
        for t in range(t_pred):
            predictions[:, t, :] = last_positions + last_velocities * (t + 1)
        
        return predictions


class NFLInference:
    """
    Inference pipeline with Test-Time Augmentation, CV Ensemble, and Smoothing.
    """
    
    def __init__(
        self,
        model_path: str,
        device: str = 'cuda',
        cv_weight: float = 0.3,  # Weight for CV baseline in ensemble
        use_tta: bool = True,
        use_smoothing: bool = True,
        smoothing_window: int = 5,
        smoothing_polyorder: int = 2
    ):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.cv_weight = cv_weight
        self.use_tta = use_tta
        self.use_smoothing = use_smoothing
        self.smoothing_window = smoothing_window
        self.smoothing_polyorder = smoothing_polyorder
        
        # Load model
        self.model = self._load_model(model_path)
        self.model.eval()
        
        print(f"Inference pipeline initialized on {self.device}")
        print(f"  TTA: {use_tta}, CV weight: {cv_weight}, Smoothing: {use_smoothing}")

    def _load_model(self, model_path: str) -> OptimizedSTGraphTransformer:
        """Load trained model from checkpoint."""
        checkpoint = torch.load(model_path, map_location=self.device)
        config = checkpoint.get('config', {})
        
        model = OptimizedSTGraphTransformer(
            input_dim=config.get('input_dim', 10),
            hidden_dim=config.get('hidden_dim', 256),
            t_obs=config.get('t_obs', 10),
            t_pred=config.get('t_pred', 50),
            nhead=config.get('nhead', 8),
            num_temporal_layers=config.get('num_temporal_layers', 2),
            num_spatial_layers=config.get('num_spatial_layers', 4),
            dropout=0.0  # No dropout at inference
        ).to(self.device)
        
        model.load_state_dict(checkpoint['model_state_dict'])
        return model

    def _flip_input(self, data: Dict[str, torch.Tensor], flip_type: int) -> Dict[str, torch.Tensor]:
        """
        Apply coordinate flips for TTA.
        flip_type: 0=none, 1=flip_x, 2=flip_y, 3=flip_both
        """
        data = {k: v.clone() if isinstance(v, torch.Tensor) else v for k, v in data.items()}
        
        if flip_type in [1, 3]:  # Flip X
            data['player_tracks'][:, :, :, 0] = 120 - data['player_tracks'][:, :, :, 0]
            data['landing_point'][:, 0] = 120 - data['landing_point'][:, 0]
            data['last_positions'][:, :, 0] = 120 - data['last_positions'][:, :, 0]
            data['ball_track'][:, :, 0] = 120 - data['ball_track'][:, :, 0]
        
        if flip_type in [2, 3]:  # Flip Y
            data['player_tracks'][:, :, :, 1] = 53.3 - data['player_tracks'][:, :, :, 1]
            data['landing_point'][:, 1] = 53.3 - data['landing_point'][:, 1]
            data['last_positions'][:, :, 1] = 53.3 - data['last_positions'][:, :, 1]
            data['ball_track'][:, :, 1] = 53.3 - data['ball_track'][:, :, 1]
        
        return data

    def _unflip_output(self, pred: torch.Tensor, flip_type: int) -> torch.Tensor:
        """Reverse the coordinate flips on predictions."""
        pred = pred.clone()
        
        if flip_type in [1, 3]:
            pred[:, :, :, 0] = 120 - pred[:, :, :, 0]
        if flip_type in [2, 3]:
            pred[:, :, :, 1] = 53.3 - pred[:, :, :, 1]
        
        return pred

    def _smooth_trajectory(self, trajectory: np.ndarray) -> np.ndarray:
        """
        Apply Savitzky-Golay filter for smooth trajectories.
        trajectory: [T, 2]
        """
        T = trajectory.shape[0]
        if T < self.smoothing_window:
            return trajectory
        
        smoothed = np.zeros_like(trajectory)
        for dim in range(2):
            smoothed[:, dim] = savgol_filter(
                trajectory[:, dim],
                window_length=self.smoothing_window,
                polyorder=self.smoothing_polyorder
            )
        return smoothed

    @torch.no_grad()
    def predict_single(self, data: Dict[str, torch.Tensor]) -> np.ndarray:
        """
        Predict trajectory for a single play with TTA and ensembling.
        
        Returns:
            predictions: [N_players, T_pred, 2] final predicted coordinates
        """
        # Move to device
        device_data = {
            k: v.to(self.device) if isinstance(v, torch.Tensor) else v
            for k, v in data.items()
        }
        
        # Neural network prediction with TTA
        nn_preds = []
        flip_types = [0, 1, 2, 3] if self.use_tta else [0]
        
        for flip_type in flip_types:
            flipped_data = self._flip_input(device_data, flip_type)
            
            pred = self.model(
                flipped_data['player_tracks'],
                flipped_data['agent_types'],
                flipped_data['last_positions'],
                flipped_data['landing_point'],
                flipped_data['ball_track']
            )
            
            # Unflip prediction
            pred = self._unflip_output(pred, flip_type)
            nn_preds.append(pred)
        
        # Average TTA predictions
        nn_pred = torch.stack(nn_preds).mean(dim=0)  # [B, N, T, 2]
        nn_pred = nn_pred.cpu().numpy()[0]  # [N, T, 2]
        
        # Constant Velocity baseline
        last_positions = data['last_positions'].numpy()[0]  # [N, 2]
        
        # Estimate velocity from last two frames
        player_tracks = data['player_tracks'].numpy()[0]  # [N, T_obs, F]
        last_velocities = player_tracks[:, -1, :2] - player_tracks[:, -2, :2]  # Assume x,y are first 2 features
        
        cv_pred = ConstantVelocityBaseline.predict(
            last_positions, last_velocities, nn_pred.shape[1]
        )
        
        # Ensemble: weighted average
        final_pred = (1 - self.cv_weight) * nn_pred + self.cv_weight * cv_pred
        
        # Apply smoothing
        if self.use_smoothing:
            for player_idx in range(final_pred.shape[0]):
                final_pred[player_idx] = self._smooth_trajectory(final_pred[player_idx])
        
        return final_pred

    def predict_batch(self, dataloader) -> List[Dict]:
        """
        Run inference on an entire dataset.
        
        Returns:
            List of dicts with game_id, play_id, and predictions
        """
        results = []
        
        for batch in tqdm(dataloader, desc="Inference"):
            B = batch['player_tracks'].shape[0]
            
            for i in range(B):
                # Extract single sample
                single_data = {
                    k: v[i:i+1] if isinstance(v, torch.Tensor) else v
                    for k, v in batch.items()
                }
                
                pred = self.predict_single(single_data)
                
                results.append({
                    'game_id': batch['game_id'][i] if isinstance(batch['game_id'], list) else batch['game_id'].item(),
                    'play_id': batch['play_id'][i] if isinstance(batch['play_id'], list) else batch['play_id'].item(),
                    'predictions': pred
                })
        
        return results

    def create_submission(self, results: List[Dict], output_path: str):
        """
        Create submission CSV in Kaggle format.
        """
        rows = []
        
        for result in results:
            game_id = result['game_id']
            play_id = result['play_id']
            preds = result['predictions']  # [N, T, 2]
            
            N, T, _ = preds.shape
            for player_idx in range(N):
                for frame_idx in range(T):
                    rows.append({
                        'gameId': game_id,
                        'playId': play_id,
                        'nflId': player_idx,  # Should be actual nflId
                        'frameId': frame_idx,
                        'x': preds[player_idx, frame_idx, 0],
                        'y': preds[player_idx, frame_idx, 1]
                    })
        
        submission = pd.DataFrame(rows)
        submission.to_csv(output_path, index=False)
        print(f"Submission saved to {output_path}")
        print(f"Shape: {submission.shape}")


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, default='./checkpoints/best.pth')
    parser.add_argument('--data_dir', type=str, default='./data')
    parser.add_argument('--output', type=str, default='./submission.csv')
    parser.add_argument('--cv_weight', type=float, default=0.3)
    parser.add_argument('--no_tta', action='store_true')
    parser.add_argument('--no_smooth', action='store_true')
    args = parser.parse_args()
    
    # Create inference pipeline
    inference = NFLInference(
        model_path=args.model_path,
        cv_weight=args.cv_weight,
        use_tta=not args.no_tta,
        use_smoothing=not args.no_smooth
    )
    
    # Load test dataset
    from dataset import NFLTrackingDataset
    from torch.utils.data import DataLoader
    
    test_dataset = NFLTrackingDataset(
        args.data_dir,
        augment=False,
        mode='test'
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=1,
        shuffle=False
    )
    
    # Run inference
    results = inference.predict_batch(test_loader)
    
    # Create submission
    inference.create_submission(results, args.output)


if __name__ == "__main__":
    main()
