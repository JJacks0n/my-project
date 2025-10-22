"""
Neural Network-based Trajectory Predictor for Sensemaking Activities

This module implements deep learning models for predicting user behavior patterns
in sensemaking trajectories using LSTM, Transformer, and other neural architectures.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
try:
    from transformers import AutoModel, AutoTokenizer
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    print("Warning: transformers not available. Some advanced features will be disabled.")
import warnings
warnings.filterwarnings('ignore')

class TrajectoryDataset(Dataset):
    """Custom dataset for trajectory sequences"""
    
    def __init__(self, sequences, labels, max_length=50):
        self.sequences = sequences
        self.labels = labels
        self.max_length = max_length
        
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        sequence = self.sequences[idx]
        label = self.labels[idx]
        
        # Pad or truncate sequence to max_length
        if len(sequence) > self.max_length:
            sequence = sequence[:self.max_length]
        else:
            # Pad with zeros
            padding = np.zeros((self.max_length - len(sequence), sequence.shape[1]))
            sequence = np.vstack([sequence, padding])
        
        return torch.FloatTensor(sequence), torch.LongTensor([label])

class LSTMTrajectoryPredictor(nn.Module):
    """LSTM-based trajectory predictor"""
    
    def __init__(self, input_size, hidden_size=128, num_layers=2, num_classes=4, dropout=0.3):
        super(LSTMTrajectoryPredictor, self).__init__()
        # Store architecture metadata for export
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_classes = num_classes
        
        # LSTM layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, 
                           batch_first=True, dropout=dropout if num_layers > 1 else 0)
        
        # Attention mechanism
        self.attention = nn.MultiheadAttention(hidden_size, num_heads=8, dropout=dropout)
        
        # Classification layers
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, num_classes)
        )
        
    def forward(self, x):
        batch_size = x.size(0)
        
        # Initialize hidden state
        h0 = torch.zeros(self.num_layers, batch_size, self.hidden_size)
        c0 = torch.zeros(self.num_layers, batch_size, self.hidden_size)
        
        # LSTM forward pass
        lstm_out, (hn, cn) = self.lstm(x, (h0, c0))
        
        # Apply attention
        attn_out, _ = self.attention(lstm_out, lstm_out, lstm_out)
        
        # Global average pooling
        pooled = torch.mean(attn_out, dim=1)
        
        # Classification
        output = self.classifier(pooled)
        
        return output

class TransformerTrajectoryPredictor(nn.Module):
    """Transformer-based trajectory predictor"""
    
    def __init__(self, input_size, d_model=128, nhead=8, num_layers=4, num_classes=4, dropout=0.1):
        super(TransformerTrajectoryPredictor, self).__init__()
        
        self.d_model = d_model
        
        # Input projection
        self.input_projection = nn.Linear(input_size, d_model)
        
        # Positional encoding
        self.pos_encoding = nn.Parameter(torch.randn(1000, d_model))
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dropout=dropout, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, num_classes)
        )
        
    def forward(self, x):
        seq_len = x.size(1)
        
        # Project input to d_model
        x = self.input_projection(x)
        
        # Add positional encoding
        x = x + self.pos_encoding[:seq_len].unsqueeze(0)
        
        # Transformer forward pass
        transformer_out = self.transformer(x)
        
        # Global average pooling
        pooled = torch.mean(transformer_out, dim=1)
        
        # Classification
        output = self.classifier(pooled)
        
        return output

class CNNTrajectoryPredictor(nn.Module):
    """CNN-based trajectory predictor for spatial patterns"""
    
    def __init__(self, input_size, num_classes=4, dropout=0.3):
        super(CNNTrajectoryPredictor, self).__init__()
        
        # 1D Convolutional layers
        self.conv1 = nn.Conv1d(input_size, 64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(64, 128, kernel_size=3, padding=1)
        self.conv3 = nn.Conv1d(128, 256, kernel_size=3, padding=1)
        
        # Batch normalization
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(256)
        
        # Pooling
        self.pool = nn.MaxPool1d(2)
        
        # Global average pooling
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes)
        )
        
    def forward(self, x):
        # Transpose for Conv1d (batch_size, channels, sequence_length)
        x = x.transpose(1, 2)
        
        # Convolutional layers
        x = torch.relu(self.bn1(self.conv1(x)))
        x = self.pool(x)
        
        x = torch.relu(self.bn2(self.conv2(x)))
        x = self.pool(x)
        
        x = torch.relu(self.bn3(self.conv3(x)))
        
        # Global average pooling
        x = self.global_pool(x)
        x = x.view(x.size(0), -1)
        
        # Classification
        output = self.classifier(x)
        
        return output

class HybridTrajectoryPredictor(nn.Module):
    """Hybrid model combining CNN, LSTM, and attention"""
    
    def __init__(self, input_size, num_classes=4, dropout=0.3):
        super(HybridTrajectoryPredictor, self).__init__()
        
        # CNN branch for spatial patterns
        self.cnn_branch = nn.Sequential(
            nn.Conv1d(input_size, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.AdaptiveAvgPool1d(1)
        )
        
        # LSTM branch for temporal patterns
        self.lstm_branch = nn.LSTM(input_size, 128, batch_first=True, dropout=dropout)
        
        # Attention mechanism
        self.attention = nn.MultiheadAttention(128, num_heads=8, dropout=dropout)
        
        # Fusion layer
        self.fusion = nn.Sequential(
            nn.Linear(128 + 128, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, num_classes)
        )
        
    def forward(self, x):
        # CNN branch
        cnn_input = x.transpose(1, 2)
        cnn_out = self.cnn_branch(cnn_input).squeeze(-1)
        
        # LSTM branch
        lstm_out, _ = self.lstm_branch(x)
        attn_out, _ = self.attention(lstm_out, lstm_out, lstm_out)
        lstm_pooled = torch.mean(attn_out, dim=1)
        
        # Fusion
        combined = torch.cat([cnn_out, lstm_pooled], dim=1)
        output = self.fusion(combined)
        
        return output

class NeuralTrajectoryTrainer:
    """Trainer class for neural trajectory models"""
    
    def __init__(self, model, device='cpu'):
        self.model = model
        self.device = device
        self.model.to(device)
        self.train_losses = []
        self.val_losses = []
        
    def train_epoch(self, dataloader, optimizer, criterion):
        """Train for one epoch"""
        self.model.train()
        total_loss = 0
        
        for batch_x, batch_y in dataloader:
            batch_x, batch_y = batch_x.to(self.device), batch_y.to(self.device)
            
            optimizer.zero_grad()
            outputs = self.model(batch_x)
            loss = criterion(outputs, batch_y.squeeze())
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        return total_loss / len(dataloader)
    
    def validate(self, dataloader, criterion):
        """Validate the model"""
        self.model.eval()
        total_loss = 0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for batch_x, batch_y in dataloader:
                batch_x, batch_y = batch_x.to(self.device), batch_y.to(self.device)
                outputs = self.model(batch_x)
                loss = criterion(outputs, batch_y.squeeze())
                
                total_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += batch_y.size(0)
                correct += (predicted == batch_y.squeeze()).sum().item()
        
        accuracy = 100 * correct / total
        return total_loss / len(dataloader), accuracy
    
    def train(self, train_loader, val_loader, epochs=50, lr=0.001, class_weights: None | torch.Tensor = None):
        """Train the model"""
        optimizer = optim.Adam(self.model.parameters(), lr=lr)
        criterion = nn.CrossEntropyLoss(weight=class_weights)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)
        
        best_val_acc = 0
        patience_counter = 0
        
        for epoch in range(epochs):
            # Training
            train_loss = self.train_epoch(train_loader, optimizer, criterion)
            self.train_losses.append(train_loss)
            
            # Validation
            val_loss, val_acc = self.validate(val_loader, criterion)
            self.val_losses.append(val_loss)
            
            # Learning rate scheduling
            scheduler.step(val_loss)
            
            # Early stopping
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                patience_counter = 0
            else:
                patience_counter += 1
                
            if patience_counter >= 10:
                print(f"Early stopping at epoch {epoch}")
                break
            
            if epoch % 10 == 0:
                print(f'Epoch {epoch}: Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%')
        
        return best_val_acc
    
    def plot_training_history(self):
        """Plot training history"""
        plt.figure(figsize=(12, 4))
        
        plt.subplot(1, 2, 1)
        plt.plot(self.train_losses, label='Training Loss')
        plt.plot(self.val_losses, label='Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.title('Training History')
        
        plt.tight_layout()
        plt.show()

def create_sequence_features(sequences):
    """Convert interaction sequences to feature matrices"""
    features = []
    
    for seq in sequences:
        if len(seq) == 0:
            continue
            
        # Convert sequence to feature matrix
        seq_features = []
        for interaction in seq:
            feature_vector = [
                interaction.get('timestamp', 0),
                hash(interaction.get('action_type', '')) % 100,  # Encode action type
                interaction.get('interaction_intensity', 0),
                interaction.get('spatial_position', [0, 0])[0],
                interaction.get('spatial_position', [0, 0])[1],
                interaction.get('data_density', 0),
                interaction.get('data_dimension', 0)
            ]
            seq_features.append(feature_vector)
        
        features.append(np.array(seq_features))
    
    return features

def main():
    """Demonstrate neural trajectory prediction"""
    print("Neural Network-based Trajectory Predictor")
    print("=" * 50)
    
    # Generate synthetic data
    from predictive_provenance_model import ProvenanceDataProcessor
    processor = ProvenanceDataProcessor()
    sequences, labels = processor.load_synthetic_data(n_samples=1000)
    
    # Convert to feature matrices
    print("Converting sequences to feature matrices...")
    feature_matrices = create_sequence_features(sequences)
    
    # Filter out empty sequences
    valid_indices = [i for i, seq in enumerate(feature_matrices) if len(seq) > 0]
    feature_matrices = [feature_matrices[i] for i in valid_indices]
    labels = [labels[i] for i in valid_indices]
    
    # Encode labels
    label_encoder = LabelEncoder()
    encoded_labels = label_encoder.fit_transform(labels)
    
    # Create datasets
    dataset = TrajectoryDataset(feature_matrices, encoded_labels, max_length=50)
    
    # Split data
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    
    # Test different models
    models = {
        'LSTM': LSTMTrajectoryPredictor(input_size=7, num_classes=len(np.unique(encoded_labels))),
        'Transformer': TransformerTrajectoryPredictor(input_size=7, num_classes=len(np.unique(encoded_labels))),
        'CNN': CNNTrajectoryPredictor(input_size=7, num_classes=len(np.unique(encoded_labels))),
        'Hybrid': HybridTrajectoryPredictor(input_size=7, num_classes=len(np.unique(encoded_labels)))
    }
    
    results = {}
    
    for model_name, model in models.items():
        print(f"\nTraining {model_name} model...")
        trainer = NeuralTrajectoryTrainer(model)
        best_acc = trainer.train(train_loader, val_loader, epochs=30)
        results[model_name] = best_acc
        
        # Plot training history
        trainer.plot_training_history()
    
    # Compare results
    print("\nModel Performance Comparison:")
    for model_name, accuracy in results.items():
        print(f"{model_name}: {accuracy:.2f}%")
    
    # Find best model
    best_model = max(results, key=results.get)
    print(f"\nBest performing model: {best_model} ({results[best_model]:.2f}%)")

if __name__ == "__main__":
    main()
