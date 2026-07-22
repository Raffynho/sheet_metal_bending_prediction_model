from pathlib import Path
from time import time

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader 

from model_piegatrice import BendingModel  
from dataset_piegatrice import load_and_preprocess_data 

base_path = Path(__file__).resolve().parent 
csv_path = "dataset_piegatura.csv" 
model_filepath = base_path / "modello_piegatrice.pth"

# Caricamento dati
print("Caricamento dati in corso...")
train_dataset, test_dataset, scaler_X, scaler_y = load_and_preprocess_data(csv_path)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

# Selezione del device di train
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
num_features = train_dataset[0][0].shape[0] 

model = BendingModel(num_features).to(device)
loss_function = nn.MSELoss() 
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Numero di epoche
epochs = 50
print(f"Training on {device}...")

training_start_time = time()

for epoch in range(epochs):
    epoch_start_time = time()
    model.train() 
    epoch_loss = 0.0
    
    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device) 
        
        # Forward pass
        outputs = model(inputs)
        loss = loss_function(outputs, labels)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        epoch_loss += loss.item()
    
    if (epoch + 1) % 10 == 0:
        epoch_end_time = time()
        epoch_time = epoch_end_time - epoch_start_time
        print(f"Epoch [{epoch+1}/{epochs}],\tLoss: {epoch_loss/len(train_loader):.4f} \t elapsed time: {epoch_time:.4f} seconds")

training_end_time = time()
print(f"Training completed in {training_end_time - training_start_time:.4f} seconds")

# Salvataggio del Modello
torch.save(model.state_dict(), model_filepath)
print(f"Model saved to {model_filepath}")