# %%

""" 
Use requirements.txt to make a virtual environment
pytorch needs to be installed separately on the venv - see the website to get the pip command

 """
import torch # type: ignore
import matplotlib.pyplot as plt # type: ignore
import pandas as pd # type: ignore
import joblib # type: ignore
import numpy as np # type: ignore
import time


# %%
device = "cuda" if torch.cuda.is_available() else "cpu"
device

# %%
from torch import nn
class NACA_NN_model_1 (nn.Module):
    def __init__(self, input_features, output_features, hidden_units=8):
        super().__init__()
        self.linear_layer_stack = nn.Sequential(
            nn.Linear(in_features=input_features, out_features=hidden_units),
            nn.ReLU(),
            nn.Linear(in_features=hidden_units, out_features=hidden_units),
            nn.ReLU(),
            nn.Linear(in_features=hidden_units, out_features=hidden_units),
            nn.ReLU(),
            nn.Linear(in_features=hidden_units, out_features=output_features)
        )

    def forward (self, x):
        return self.linear_layer_stack(x)

# %%
path_to_model_data = "models/model_2/model_2.pth"
model_data = torch.load(path_to_model_data)

config = model_data["config"]
model_2 = NACA_NN_model_1(**config).to(device)

model_2.load_state_dict(model_data["model_state"])


# %%
# Put the model in a wrapper class
from helper_functions import ModelWrapper
model_2_input_scaler = joblib.load("models/model_2/input_scaler.pkl")
model_2_output_scaler = joblib.load("models/model_2/output_scaler.pkl")
predictor = ModelWrapper(model_2, model_2_input_scaler, model_2_output_scaler, device)

# %%
# plot change in reynolds num
camber = 6
camber_pos = 0.8
thickness = 10
Re = 500000
mach = .2
alpha = 8

input_data = [[camber, camber_pos, thickness, Re, mach, alpha]]

prediction = predictor.predict(input_data)
prediction


# %%
results = []
prediction_count = 0
start_time = time.time()
for alpha in np.arange(0, 8.001, 0.01):
    input_data[0][5] = alpha
    result = predictor.predict(input_data)
    results.append(result)
    prediction_count += 1
end_time = time.time()
total_time = end_time - start_time
print(f"{prediction_count} predictions in {total_time} seconds\n{total_time / prediction_count} seconds per prediction")
results

# %%

# Stack into a 2D array
data = np.vstack(results)  # shape: (n_samples, 3)

# Optional: alpha values if known, else just use index
alpha_values = np.linspace(0, 8, len(data))  # Assuming alpha varied from 0 to 8

# Plot CL
plt.figure(figsize=(6, 4))
plt.plot(alpha_values, data[:, 0], marker='o', label="CL")
plt.xlabel("Alpha (°)")
plt.ylabel("CL")
plt.title("Predicted CL vs Alpha")
plt.grid(True)
plt.tight_layout()
plt.show()
plt.savefig("eval_graphs/NN_CL.png")

# Plot CD
plt.figure(figsize=(6, 4))
plt.plot(alpha_values, data[:, 1], marker='o', color='red', label="CD")
plt.xlabel("Alpha (°)")
plt.ylabel("CD")
plt.title("Predicted CD vs Alpha")
plt.grid(True)
plt.tight_layout()
plt.show()
plt.savefig("eval_graphs/NN_CD.png")

# Plot CM
plt.figure(figsize=(6, 4))
plt.plot(alpha_values, data[:, 2], marker='o', color='green', label="CM")
plt.xlabel("Alpha (°)")
plt.ylabel("CM")
plt.title("Predicted CM vs Alpha")
plt.grid(True)
plt.tight_layout()
plt.show()
plt.savefig("eval_graphs/NN_CM.png")


# %%
df = pd.read_csv("naca_sim_data.csv")

# Filter data
filtered_df = df[
    (df["Camber (%)"] == camber) &
    (df["Camber Position (x/c)"] == camber_pos) &
    (df["Thickness (%)"] == thickness) &
    (df["Re"] == Re) &
    (df["Mach"] == mach)
].sort_values(by="Alpha")

# Alpha and outputs
alpha = filtered_df["Alpha"]
cl = filtered_df["CL"]
cd = filtered_df["CD"]
cm = filtered_df["CM"]


# %%

plt.figure(figsize=(6, 4))
plt.plot(alpha, cl, marker='o', label="CL")
plt.xlabel("Alpha (°)")
plt.ylabel("CL")
plt.title("Lift Coefficient vs Alpha")
plt.grid(True)
plt.tight_layout()
plt.show()
plt.savefig("eval_graphs/xfoil_CL.png")

plt.figure(figsize=(6, 4))
plt.plot(alpha, cd, marker='o', color='red', label="CD")
plt.xlabel("Alpha (°)")
plt.ylabel("CD")
plt.title("Drag Coefficient vs Alpha")
plt.grid(True)
plt.tight_layout()
plt.show()
plt.savefig("eval_graphs/xfoil_CD.png")


plt.figure(figsize=(6, 4))
plt.plot(alpha, cm, marker='o', color='green', label="CM")
plt.xlabel("Alpha (°)")
plt.ylabel("CM")
plt.title("Moment Coefficient vs Alpha")
plt.grid(True)
plt.tight_layout()
plt.show()
plt.savefig("eval_graphs/xfoil_CM.png")


# %%



