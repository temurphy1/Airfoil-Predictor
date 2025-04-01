import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# Define a custom Dataset class for handling the NACA airfoil data
class NACA_Dataset(Dataset):
    def __init__(self, features, targets):
        """
        Initialize the dataset
        Args:
            features: NACA airfoil numbers (input features)
            targets: Aerodynamic coefficients (Cl, Cd, Cm)
        """
        self.features = torch.FloatTensor(features)
        self.targets = torch.FloatTensor(targets)
    
    def __len__(self):
        return len(self.features)
    
    def __getitem__(self, idx):
        return self.features[idx], self.targets[idx]

# Define the Neural Network architecture
class NACA_Net(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        """
        Initialize the neural network
        Args:
            input_size: Size of input features (NACA number)
            hidden_size: Number of neurons in hidden layers
            output_size: Number of output features (Cl, Cd, Cm)
        """
        super(NACA_Net, self).__init__()
        self.layer1 = nn.Linear(input_size, hidden_size)
        self.layer2 = nn.Linear(hidden_size, hidden_size)
        self.layer3 = nn.Linear(hidden_size, output_size)
        
        # Activation functions
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.2)  # Dropout for regularization
    
    def forward(self, x):
        """
        Forward pass through the network
        Args:
            x: Input features
        Returns:
            Predicted aerodynamic coefficients
        """
        x = self.relu(self.layer1(x))
        x = self.dropout(x)
        x = self.relu(self.layer2(x))
        x = self.dropout(x)
        x = self.layer3(x)
        return x

def load_and_preprocess_data(csv_path):
    """
    Load and preprocess the data from CSV file
    Args:
        csv_path: Path to the CSV file containing the data
    Returns:
        Preprocessed features and targets
    """
    # Load the CSV file
    df = pd.read_csv(csv_path)
    
    # Extract features (NACA numbers) and targets (Cl, Cd, Cm)
    features = df['NACA'].values
    targets = df[['Cl', 'Cd', 'Cm']].values
    
    # Normalize the features and targets
    features_mean = np.mean(features)
    features_std = np.std(features)
    features_normalized = (features - features_mean) / features_std
    
    targets_mean = np.mean(targets, axis=0)
    targets_std = np.std(targets, axis=0)
    targets_normalized = (targets - targets_mean) / targets_std
    
    return features_normalized, targets_normalized, features_mean, features_std, targets_mean, targets_std

def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs, device):
    """
    Train the neural network
    Args:
        model: The neural network model
        train_loader: DataLoader for training data
        val_loader: DataLoader for validation data
        criterion: Loss function
        optimizer: Optimizer
        num_epochs: Number of training epochs
        device: Device to train on (CPU/GPU)
    """
    train_losses = []
    val_losses = []
    
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        train_loss = 0
        for features, targets in train_loader:
            features, targets = features.to(device), targets.to(device)
            
            optimizer.zero_grad()
            outputs = model(features)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
        
        # Validation phase
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for features, targets in val_loader:
                features, targets = features.to(device), targets.to(device)
                outputs = model(features)
                val_loss += criterion(outputs, targets).item()
        
        train_losses.append(train_loss / len(train_loader))
        val_losses.append(val_loss / len(val_loader))
        
        if (epoch + 1) % 10 == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}], '
                  f'Train Loss: {train_losses[-1]:.4f}, '
                  f'Val Loss: {val_losses[-1]:.4f}')
    
    return train_losses, val_losses

def main():
    # Set random seed for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    
    # Check if GPU is available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load and preprocess data
    csv_path = "naca_data.csv"  # Update this path to your CSV file
    features, targets, features_mean, features_std, targets_mean, targets_std = load_and_preprocess_data(csv_path)
    
    # Split data into training and validation sets
    X_train, X_val, y_train, y_val = train_test_split(features, targets, test_size=0.2, random_state=42)
    
    # Create datasets and dataloaders
    train_dataset = NACA_Dataset(X_train, y_train)
    val_dataset = NACA_Dataset(X_val, y_val)
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32)
    
    # Initialize model, loss function, and optimizer
    input_size = 1  # NACA number
    hidden_size = 64
    output_size = 3  # Cl, Cd, Cm
    
    model = NACA_Net(input_size, hidden_size, output_size).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # Train the model
    num_epochs = 100
    train_losses, val_losses = train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs, device)
    
    # Plot training and validation losses
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Losses')
    plt.legend()
    plt.show()
    
    # Save the trained model
    torch.save({
        'model_state_dict': model.state_dict(),
        'features_mean': features_mean,
        'features_std': features_std,
        'targets_mean': targets_mean,
        'targets_std': targets_std
    }, 'naca_model.pth')
    
    print("Model saved successfully!")

if __name__ == "__main__":
    main()
