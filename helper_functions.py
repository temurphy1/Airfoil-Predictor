import torch
class ModelWrapper:
    def __init__(self, model, input_scaler, output_scaler, device):
        self.model = model
        self.input_scaler = input_scaler
        self.output_scaler = output_scaler
        self.device = device

    def predict(self, X):
        X_scaled = self.input_scaler.transform(X)
        X_tensor = torch.tensor(X_scaled, dtype=torch.float32).to(self.device)

        with torch.inference_mode():
            y_scaled = self.model(X_tensor).cpu().numpy()

        return self.output_scaler.inverse_transform(y_scaled)
