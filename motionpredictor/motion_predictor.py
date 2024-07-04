import torch
from torch import nn
from diffusers import DiffusionPipeline
from diffusers.configuration_utils import ConfigMixin
from diffusers.modeling_utils import ModelMixin

class MotionPredictor(ModelMixin, ConfigMixin):
    def __init__(self, hidden_dim=1024, num_heads=12, num_layers=8, config=None):
        super().__init__()
        self.config = config
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=num_heads),
            num_layers=num_layers
        )
        self.output_projection = nn.Linear(hidden_dim, hidden_dim)  # Adjust the dimension if necessary

    def predict_motion(self, start_tokens, end_tokens):
        # Linear interpolation in the token space
        interpolation_steps = torch.linspace(0, 1, steps=10, device=start_tokens.device)[:, None, None]
        interpolated_tokens = start_tokens[None, :] * (1 - interpolation_steps) + end_tokens[None, :] * interpolation_steps
        
        # Transformer predicts the motion along frame dimension
        motion_tokens = self.transformer(interpolated_tokens.squeeze(0))
        motion_tokens = self.output_projection(motion_tokens)  # Final dimension projection
        return motion_tokens

    def forward(self, start_tokens, end_tokens):
        return self.predict_motion(start_tokens, end_tokens)
