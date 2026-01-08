"""
NFL Big Data Bowl 2026 - Optimized ST-GraphTransformer
Based on 1st-5th place winning solution techniques.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding for sequence data."""
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

class TemporalEncoderGRU(nn.Module):
    """
    Bidirectional GRU for temporal encoding (winner technique).
    Input: [Batch, T_obs, n_features]
    Output: [Batch, hidden_dim]
    """
    def __init__(self, input_dim, hidden_dim, num_layers=2, dropout=0.1):
        super().__init__()
        self.gru = nn.GRU(
            input_dim, hidden_dim // 2,  # Bidirectional doubles output
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0
        )
        self.layer_norm = nn.LayerNorm(hidden_dim)

    def forward(self, x):
        # x: [Batch, T_obs, Feat]
        out, _ = self.gru(x)  # [Batch, T_obs, Hidden]
        # Take last timestep output (from both directions)
        last_out = out[:, -1, :]  # [Batch, Hidden]
        return self.layer_norm(last_out)

class SpatialInteractionLayer(nn.Module):
    """
    Transformer Encoder for spatial interactions.
    Key: Ball landing point treated as 24th agent.
    """
    def __init__(self, hidden_dim, nhead=8, num_layers=3, dropout=0.1):
        super().__init__()
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=nhead,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            batch_first=True,
            activation='gelu'
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.layer_norm = nn.LayerNorm(hidden_dim)

    def forward(self, src, src_key_padding_mask=None):
        out = self.transformer(src, src_key_padding_mask=src_key_padding_mask)
        return self.layer_norm(out)

class TrajectoryDecoder(nn.Module):
    """
    Non-autoregressive decoder predicting all future frames at once.
    Predicts velocity (dx, dy) then integrates for position.
    """
    def __init__(self, hidden_dim, t_pred, dropout=0.1):
        super().__init__()
        self.t_pred = t_pred
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, t_pred * 2)  # (dx, dy) per frame
        )

    def forward(self, x):
        out = self.mlp(x)
        return out.view(-1, self.t_pred, 2)

class OptimizedSTGraphTransformer(nn.Module):
    """
    Optimized Hybrid Spatial-Temporal Graph Transformer.
    
    Key optimizations from winning solutions:
    1. Ball landing point as additional "agent" in Transformer
    2. Bidirectional GRU temporal encoder
    3. Multi-head attention with GELU activation
    4. Layer normalization for stability
    5. Predicts velocity (dx, dy) for smoother trajectories
    """
    def __init__(
        self,
        input_dim=10,
        hidden_dim=256,
        t_obs=10,
        t_pred=50,
        num_agents=22,  # 22 players (11 offense + 11 defense)
        include_ball=True,
        include_landing=True,  # KEY: Ball landing as agent
        nhead=8,
        num_temporal_layers=2,
        num_spatial_layers=4,
        dropout=0.1
    ):
        super().__init__()
        self.num_agents = num_agents
        self.include_ball = include_ball
        self.include_landing = include_landing
        self.hidden_dim = hidden_dim
        self.t_pred = t_pred
        
        # Total agents: players + ball + landing point
        total_agents = num_agents
        if include_ball:
            total_agents += 1
        if include_landing:
            total_agents += 1
        self.total_agents = total_agents
        
        # Input projection
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        
        # Temporal Encoder (shared)
        self.temporal_encoder = TemporalEncoderGRU(
            hidden_dim, hidden_dim,
            num_layers=num_temporal_layers,
            dropout=dropout
        )
        
        # Agent type embeddings (0: Offense, 1: Defense, 2: Ball, 3: Landing)
        self.type_embedding = nn.Embedding(4, hidden_dim)
        
        # Positional encoding for agent ordering (optional)
        self.pos_encoding = PositionalEncoding(hidden_dim, max_len=total_agents)
        
        # Spatial Transformer
        self.spatial_transformer = SpatialInteractionLayer(
            hidden_dim, nhead=nhead,
            num_layers=num_spatial_layers,
            dropout=dropout
        )
        
        # Decoder (per agent)
        self.decoder = TrajectoryDecoder(hidden_dim, t_pred, dropout=dropout)
        
        # Last position embedding (for relative prediction)
        self.last_pos_proj = nn.Linear(2, hidden_dim)

    def forward(self, player_tracks, agent_types, last_positions, landing_point=None, ball_track=None):
        """
        Args:
            player_tracks: [B, 22, T_obs, F] - Player tracking data
            agent_types: [B, 22] - Agent type IDs (0: Off, 1: Def)
            last_positions: [B, 22, 2] - Last known (x, y) before prediction
            landing_point: [B, 2] - Ball landing coordinates (target location)
            ball_track: [B, T_obs, F] - Ball tracking data (optional)
        
        Returns:
            preds: [B, N, T_pred, 2] - Predicted (x, y) coordinates
        """
        B, N, T, F = player_tracks.shape
        device = player_tracks.device
        
        agents_list = []
        types_list = []
        last_pos_list = []
        
        # === Process Players ===
        # Project to hidden dim
        x = self.input_proj(player_tracks)  # [B, N, T, H]
        x_flat = x.view(B * N, T, -1)
        h_players = self.temporal_encoder(x_flat)  # [B*N, H]
        h_players = h_players.view(B, N, -1)  # [B, N, H]
        agents_list.append(h_players)
        types_list.append(agent_types)
        last_pos_list.append(last_positions)
        
        # === Process Ball (if included) ===
        if self.include_ball and ball_track is not None:
            ball_x = self.input_proj(ball_track)  # [B, T, H]
            h_ball = self.temporal_encoder(ball_x)  # [B, H]
            agents_list.append(h_ball.unsqueeze(1))  # [B, 1, H]
            ball_type = torch.full((B, 1), 2, dtype=torch.long, device=device)
            types_list.append(ball_type)
            ball_last_pos = ball_track[:, -1, :2]  # Assume first 2 features are x, y
            last_pos_list.append(ball_last_pos.unsqueeze(1))
        
        # === Process Landing Point (KEY OPTIMIZATION) ===
        if self.include_landing and landing_point is not None:
            # Landing point doesn't have temporal history - create static embedding
            landing_embed = torch.zeros(B, self.hidden_dim, device=device)
            # Encode landing point through a simple projection
            landing_xy = landing_point  # [B, 2]
            landing_proj = self.last_pos_proj(landing_xy)  # [B, H]
            landing_embed = landing_embed + landing_proj
            agents_list.append(landing_embed.unsqueeze(1))  # [B, 1, H]
            landing_type = torch.full((B, 1), 3, dtype=torch.long, device=device)
            types_list.append(landing_type)
            last_pos_list.append(landing_point.unsqueeze(1))
        
        # === Concatenate all agents ===
        h_all = torch.cat(agents_list, dim=1)  # [B, Total_Agents, H]
        all_types = torch.cat(types_list, dim=1)  # [B, Total_Agents]
        all_last_pos = torch.cat(last_pos_list, dim=1)  # [B, Total_Agents, 2]
        
        # Add type embeddings
        type_embed = self.type_embedding(all_types)  # [B, Total_Agents, H]
        h_all = h_all + type_embed
        
        # Add positional encoding
        h_all = self.pos_encoding(h_all)
        
        # Add last position embedding
        last_pos_embed = self.last_pos_proj(all_last_pos)  # [B, Total_Agents, H]
        h_all = h_all + last_pos_embed
        
        # === Spatial Transformer ===
        h_spatial = self.spatial_transformer(h_all)  # [B, Total_Agents, H]
        
        # === Decode trajectories ===
        total_agents = h_spatial.size(1)
        h_flat = h_spatial.view(B * total_agents, -1)
        delta_preds = self.decoder(h_flat)  # [B*Total, T_pred, 2]
        delta_preds = delta_preds.view(B, total_agents, self.t_pred, 2)
        
        # Convert velocity to position by cumulative sum
        preds = torch.cumsum(delta_preds, dim=2)  # Integrate velocity
        preds = preds + all_last_pos.unsqueeze(2)  # Add last known position
        
        # Return only player predictions (exclude ball and landing)
        return preds[:, :N, :, :]

if __name__ == "__main__":
    # Smoke Test
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Testing on device: {device}")
    
    model = OptimizedSTGraphTransformer(
        input_dim=10,
        hidden_dim=128,
        t_obs=10,
        t_pred=50,
        num_agents=22
    ).to(device)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    # Test forward pass
    B = 4
    player_tracks = torch.randn(B, 22, 10, 10).to(device)
    agent_types = torch.randint(0, 2, (B, 22)).to(device)
    last_positions = torch.randn(B, 22, 2).to(device)
    landing_point = torch.randn(B, 2).to(device)
    ball_track = torch.randn(B, 10, 10).to(device)
    
    output = model(player_tracks, agent_types, last_positions, landing_point, ball_track)
    print(f"Output shape: {output.shape}")  # Should be [4, 22, 50, 2]
    print("✓ Smoke test passed!")
