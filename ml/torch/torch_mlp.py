import colorama
import numpy as np
from ml.preprocess import DATASET_TYPE
from tqdm import tqdm

try:
    import torch
except ModuleNotFoundError as e:
    print(colorama.Style.DIM + colorama.Fore.RED + "PyTorch installation not found.")
    print(
        colorama.Fore.RESET + "To fix, install PyTorch. See README.md for instructions."
    )
    exit(1)

import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader


class DatasetBSOR(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.y)

    def __getitem__(self, index):
        x = self.X[index]
        y = self.y[index].astype(np.long)
        return {
            "data": torch.tensor(x, dtype=torch.float32),
            "label": torch.tensor(y, dtype=torch.long),
        }


class MLPclassifier(nn.Module):
    def __init__(self, d_input, d_hidden):
        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(d_input, d_hidden),
            nn.ReLU(),
            nn.Linear(d_hidden, d_hidden),
            nn.ReLU(),
            nn.Linear(d_hidden, 2),
        )

    def forward(self, x):
        return self.network(x)


def train_model(dataset: DATASET_TYPE) -> str:
    """
    Train basic model using Torch MLP classifier. Exports as `mlp_torch.onnx`.
    """
    print("  -> Training Torch MLP classifier")

    X, X_test, y, y_test = dataset

    d_input = len(X[0])
    d_hidden = 32
    model = MLPclassifier(d_input, d_hidden)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"    -> Using device: {device}\n")

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    batch_size = 32
    num_workers = 1
    epochs = 10

    train_dataset = DatasetBSOR(X, y)
    test_dataset = DatasetBSOR(X_test, y_test)

    train_loader = DataLoader(
        dataset=train_dataset,
        shuffle=True,
        batch_size=batch_size,
        num_workers=num_workers,
    )

    test_loader = DataLoader(
        dataset=test_dataset,
        shuffle=False,
        batch_size=batch_size,
        num_workers=num_workers,
    )

    model = model.to(device)

    model.train()
    for epoch in range(epochs):
        epoch_loss = 0.0

        for batch in tqdm(
            train_loader,
            total=len(train_loader),
            desc=f"Epoch {epoch+1}/{epochs}",
            leave=True,
        ):
            batch = {k: v.to(device) for k, v in batch.items()}
            optimizer.zero_grad()

            data = batch["data"]
            labels = batch["label"]

            outputs = model(data)
            loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        print(f"Epoch {epoch+1} Average Loss: {epoch_loss / len(train_loader):.4f}\n")

    model.eval()
    with torch.no_grad():
        test_loss = 0.0

        for batch in tqdm(
            test_loader, total=len(test_loader), desc=f"Evaluation", leave=True
        ):
            batch = {k: v.to(device) for k, v in batch.items()}

            data = batch["data"]
            labels = batch["label"]

            outputs = model(data)
            loss = criterion(outputs, labels)

            test_loss += loss.item()

        print(f"Test set, Average Loss: {test_loss / len(test_loader):.4f}\n")

    # EXPORT
    print("  --> Exporting model")
    if torch.cuda.is_available():
        model = model.to("cpu")
    FILE = "models/mlp_torch.onnx"
    torch.onnx.export(model, (torch.tensor(X, dtype=torch.float32),), FILE, dynamo=True)
    print("  --> Training complete\n")

    return FILE
