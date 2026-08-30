import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional
import yaml


class CNNAutoencoder(nn.Module):
    def __init__(self, latent_dim=128, in_channels=1, input_shape=(128, 313)):
        super().__init__()
        self.input_shape = input_shape
        
        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, 32, 3, stride=2, padding=1),  # 64x157
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout2d(0.3),
            
            nn.Conv2d(32, 64, 3, stride=2, padding=1),  # 32x79
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout2d(0.3),
            
            nn.Conv2d(64, 128, 3, stride=2, padding=1),  # 16x40
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout2d(0.3),
            
            nn.Conv2d(128, 256, 3, stride=2, padding=1),  # 8x20
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
        )
        
        self.adaptive_pool = nn.AdaptiveAvgPool2d((4, 4))
        self.fc_mu = nn.Linear(256 * 4 * 4, latent_dim)
        
        # Decoder
        self.fc_dec = nn.Linear(latent_dim, 256 * 4 * 4)
        
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(256, 128, 3, stride=2, padding=1, output_padding=1),  # 4x4 -> 8x8
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.ConvTranspose2d(128, 64, 3, stride=2, padding=1, output_padding=1),  # 8x8 -> 16x16
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.ConvTranspose2d(64, 32, 3, stride=2, padding=1, output_padding=1),  # 16x16 -> 32x32
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.ConvTranspose2d(32, in_channels, 3, stride=2, padding=1, output_padding=1),  # 32x32 -> 64x64
            nn.Sigmoid()
        )
        
        # Final upsample to match input
        self.final_upsample = nn.Upsample(size=input_shape, mode='bilinear', align_corners=False)
    
    def encode(self, x):
        h = self.encoder(x)
        h = self.adaptive_pool(h)
        h = h.view(h.size(0), -1)
        return self.fc_mu(h)
    
    def decode(self, z):
        h = self.fc_dec(z)
        h = h.view(h.size(0), 256, 4, 4)
        h = self.decoder(h)
        return self.final_upsample(h)
    
    def forward(self, x):
        z = self.encode(x)
        return self.decode(z), z
    
    def get_reconstruction_error(self, x):
        with torch.no_grad():
            recon, _ = self.forward(x)
            error = torch.mean((x - recon) ** 2, dim=[1, 2, 3])
        return error


class LSTMAutoencoder(nn.Module):
    def __init__(
        self,
        input_dim: int = 128,
        hidden_dim: int = 256,
        latent_dim: int = 128,
        num_layers: int = 2,
        dropout: float = 0.3
    ):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        
        self.encoder_lstm = nn.LSTM(
            input_dim, hidden_dim, num_layers,
            batch_first=True, dropout=dropout, bidirectional=True
        )
        
        self.encoder_fc = nn.Linear(hidden_dim * 2, latent_dim)
        
        self.decoder_fc = nn.Linear(latent_dim, hidden_dim)
        
        self.decoder_lstm = nn.LSTM(
            hidden_dim, hidden_dim, num_layers,
            batch_first=True, dropout=dropout, bidirectional=False
        )
        
        self.output_fc = nn.Linear(hidden_dim, input_dim)
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_size, channels, freq, time = x.shape
        x = x.permute(0, 3, 1, 2).contiguous().view(batch_size, time, -1)
        
        _, (hidden, _) = self.encoder_lstm(x)
        hidden = hidden[-2:].permute(1, 0, 2).contiguous().view(batch_size, -1)
        latent = self.encoder_fc(hidden)
        
        decoded = self.decoder_fc(latent).unsqueeze(1).repeat(1, time, 1)
        decoded, _ = self.decoder_lstm(decoded)
        reconstructed = self.output_fc(decoded)
        
        reconstructed = reconstructed.view(batch_size, time, channels, freq).permute(0, 2, 3, 1)
        
        return reconstructed, latent


def create_model(config: dict) -> nn.Module:
    model_config = config['training']['model']
    model_type = model_config['architecture']
    
    if model_type == "cnn_autoencoder":
        return CNNAutoencoder(
            in_channels=1,
            latent_dim=model_config['latent_dim'],
            input_shape=(config['data']['audio']['n_mels'], config['data']['audio']['n_frames'])
        )
    elif model_type == "lstm_autoencoder":
        return LSTMAutoencoder(
            input_dim=config['data']['audio']['n_mels'],
            latent_dim=model_config['latent_dim'],
            dropout=model_config['dropout']
        )
    else:
        raise ValueError(f"Unknown model architecture: {model_type}")


def load_model(model_path: str, config: dict, device: torch.device) -> nn.Module:
    model = create_model(config)
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    return model


if __name__ == "__main__":
    with open("config.yaml", "r") as f:
        config = yaml.safe_load(f)
    
    device = torch.device(config['training']['device'] if torch.cuda.is_available() else "cpu")
    model = create_model(config).to(device)
    
    x = torch.randn(2, 1, 128, 313).to(device)
    reconstructed, latent = model(x)
    
    print(f"Input shape: {x.shape}")
    print(f"Latent shape: {latent.shape}")
    print(f"Reconstructed shape: {reconstructed.shape}")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")