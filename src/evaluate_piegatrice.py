from pathlib import Path
import torch
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from model_piegatrice import BendingModel
from dataset_piegatrice import load_and_preprocess_data

base_path = Path(__file__).resolve().parent
csv_path = "dataset_piegatura.csv" 
model_filepath = base_path / "modello_piegatrice.pth"

_, test_dataset, _, scaler_y = load_and_preprocess_data(csv_path)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
num_features = test_dataset[0][0].shape[0]

model = BendingModel(num_features).to(device)
model.load_state_dict(torch.load(model_filepath, map_location=device))
model.eval()

all_preds = []
all_targets = []

with torch.no_grad():
    for inputs, targets in test_loader:
        inputs, targets = inputs.to(device), targets.to(device)
        outputs = model(inputs)
        
        all_preds.extend(outputs.cpu().numpy())
        all_targets.extend(targets.cpu().numpy())

# 3. Denormalizzazione
all_preds_real = scaler_y.inverse_transform(all_preds)
all_targets_real = scaler_y.inverse_transform(all_targets)

# 4. Calcolo Errori e Residui
mae = mean_absolute_error(all_targets_real, all_preds_real)
mse = mean_squared_error(all_targets_real, all_preds_real)
r2 = r2_score(all_targets_real, all_preds_real)

# Calcolo vettoriale dei residui
residui = all_preds_real - all_targets_real

print(f"--- RISULTATI DELLA VALUTAZIONE ---")
print(f'Errore Assoluto Medio (MAE): {mae:.4f} mm')
print(f'Errore Quadratico Medio (MSE): {mse:.4f}')
print(f'R2 Score: {r2:.4f}')

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

ax1.scatter(all_targets_real, all_preds_real, alpha=0.6, color='blue')

min_val = min(all_targets_real.min(), all_preds_real.min())
max_val = max(all_targets_real.max(), all_preds_real.max())
ax1.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Previsione Perfetta') 

testo_metriche = f'MAE: {mae:.4f} mm\nMSE: {mse:.4f}\nR2: {r2:.4f}'
ax1.text(0.05, 0.85, testo_metriche, transform=ax1.transAxes, fontsize=12, 
         bbox=dict(boxstyle="round,pad=0.5", facecolor="white", edgecolor="gray", alpha=0.8))

ax1.set_xlabel('Movimento Reale (Target) [mm]')
ax1.set_ylabel('Movimento Previsto dal Modello [mm]')
ax1.set_title('Valutazione Modello: Reale vs Previsto')
ax1.legend(loc='lower right')
ax1.grid(True, linestyle='--', alpha=0.7)

ax2.scatter(all_targets_real, residui, alpha=0.6, color='red', marker='x')

ax2.axhline(y=0, color='black', linestyle='--', linewidth=2, label='Errore Zero')

ax2.set_xlabel('Movimento Reale (Target) [mm]')
ax2.set_ylabel('Residuo / Errore (Previsto - Reale) [mm]')
ax2.set_title('Analisi dei Residui: Diagnostica Modello')
ax2.legend()
ax2.grid(True, linestyle='--', alpha=0.7)

# Mostra il tutto
plt.tight_layout()
plt.show()
