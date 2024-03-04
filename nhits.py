import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset


class NHiTS_Dataset(Dataset):
    def __init__(self, series, backcast_size, forecast_size):
        self.series = series
        self.backcast_size = backcast_size
        self.forecast_size = forecast_size

    def __len__(self):
        return len(self.series) - self.backcast_size - self.forecast_size + 1

    def __getitem__(self, idx):
        input_data = self.series[idx : idx + self.backcast_size]
        target_data = self.series[
            idx + self.backcast_size : idx + self.backcast_size + self.forecast_size
        ]

        # Convert to PyTorch tensors
        input_data = torch.Tensor(input_data.values)
        target_data = torch.Tensor(target_data.values)

        return input_data, target_data


class MLP(nn.Module):
    def __init__(self, hidden_size):
        super(MLP, self).__init__()
        self.layer = nn.Sequential(
            nn.Linear(hidden_size, hidden_size, bias=False),
            nn.ReLU(),
            nn.LayerNorm(hidden_size),
        )

    def forward(self, x):
        return self.layer(x)


class InterpolationBasis(nn.Module):
    def __init__(self, backcast_size, forecast_size):
        self.backcast_size = backcast_size
        self.forecast_size = forecast_size

    def forward(self, bcc, fcc):
        backcast = F.interpolate(bcc.unsqueeze(1), self.backcast_size).squeeze()
        forecast = F.interpolate(fcc.unsqueeze(1), self.forecast_size).squeeze()
        return backcast, forecast


class NHiTS_Block_Redesign(nn.Module):
    def __init__(
        self,
        pooling_size,
        pooling_mode,
        mlp_hidden_size,
        mlp_num_layers,
        bcc_size,
        fcc_size,
        backcast_size,
    ):
        super().__init__()
        assert pooling_mode in ["max", "avg"], "pooling_mode must be max or avg"

        self.bcc_size = bcc_size
        self.fcc_size = fcc_size

        self.pooling = (
            nn.MaxPool1d(kernel_size=pooling_size, stride=pooling_size)
            if pooling_mode == "max"
            else nn.AvgPool1d(kernel_size=pooling_size, stride=pooling_size)
        )
        self.mlps = nn.Sequential(
            nn.Linear(backcast_size // pooling_size, mlp_hidden_size),
            *(MLP(mlp_hidden_size) for _ in range(mlp_num_layers)),
        )

        self.bcc_linear = nn.Linear(mlp_hidden_size, bcc_size)
        self.fcc_linear = nn.Linear(mlp_hidden_size, fcc_size)

    def forward(self, x):
        x = self.pooling(x)
        x = self.mlps(x)

        bcc = self.bcc_linear(x)
        fcc = self.fcc_linear(x)

        return bcc, fcc


class NHiTS_Stack_Redesign(nn.Module):
    def __init__(
        self,
        num_blocks,
        pooling_size,
        pooling_mode,
        mlp_hidden_size,
        mlp_num_layers,
        bcc_size,
        fcc_size,
        backcast_size,
        forecast_size,
    ):
        super().__init__()

        self.backcast_size = backcast_size
        self.forecast_size = forecast_size

        self.basis = InterpolationBasis(
            backcast_size=backcast_size, forecast_size=forecast_size
        )

        self.blocks = nn.ModuleList(
            [
                NHiTS_Block_Redesign(
                    pooling_size=pooling_size[i],
                    pooling_mode=pooling_mode,
                    mlp_hidden_size=mlp_hidden_size,
                    mlp_num_layers=mlp_num_layers,
                    bcc_size=bcc_size,
                    fcc_size=fcc_size,
                    backcast_size=backcast_size,
                )
                for i in range(num_blocks)
            ]
        )

    def forward(self, x):
        res_stream = x
        outputs = []
        for block in self.blocks:
            bcc, fcc = block(res_stream)

            backcast = F.interpolate(bcc.unsqueeze(1), self.backcast_size).squeeze()
            self.forecast_logger.log_forecasts()
            forecast = F.interpolate(fcc.unsqueeze(1), self.forecast_size).squeeze()
            outputs.append(forecast)
            res_stream = res_stream - backcast

        return torch.sum(torch.stack(outputs), dim=0)


class NHiTS_Redesign(nn.Module):
    def __init__(
        self,
        stacks_num_layers,
        backcast_size,
        forecast_size,
        num_blocks,
        pooling_size,
        pooling_mode,
        mlp_hidden_size,
        mlp_num_layers,
        bcc_size,
        fcc_size,
    ):
        super().__init__()

        self.forecast_size = forecast_size
        self.backcast_size = backcast_size
        # TODO: FIT LOGGING INITIALIZATION
        self.stacks = nn.ModuleList(
            [
                NHiTS_Stack_Redesign(
                    backcast_size=backcast_size,
                    forecast_size=forecast_size,
                    num_blocks=num_blocks,
                    pooling_size=pooling_size[i],
                    pooling_mode=pooling_mode,
                    mlp_hidden_size=mlp_hidden_size,
                    mlp_num_layers=mlp_num_layers,
                    bcc_size=bcc_size[i],
                    fcc_size=fcc_size[i],
                )
                for i in range(stacks_num_layers)
            ]
        )

    def forward(self, x):
        res_stream = x
        outputs = []
        for stack in self.stacks:
            outputs.append(stack(res_stream))
        return torch.sum(torch.stack(outputs), dim=0)

    def fit(self, dataloader, loss_func, lr, optim, num_epochs):
        optimizer = optim(self.parameters(), lr=lr)

        counter = 0
        for epoch in range(num_epochs):
            for batch_idx, (data, target) in enumerate(dataloader):
                # Forward pass
                output = self.forward(data)
                # Compute loss
                loss = loss_func(output, target)

                # Backward pass and optimization
                optimizer.zero_grad()
                loss.backward()

                optimizer.step()
                counter += 1

                print(
                    f"Epoch [{epoch+1}/{num_epochs}], Step [{batch_idx+1}/{len(dataloader)}], Loss: {loss.item()}"
                )

    def predict(self, data_loader):
        self.eval()
        predictions = []
        with torch.no_grad():
            for data in data_loader:
                output = self.forward(data)
                predictions.append(output)
        return torch.stack(predictions)

    def predict_single(self, sample):
        assert sample.shape[-1] == self.backcast_size and len(
            sample.shape == 2
        ), "Must have rank 2 and final shape dim be len backcast_size-> (1, backcast_size)"
        self.eval()
        predictions = []
        with torch.no_grad():
            output = self.forward(sample)
            predictions.append(output)
        return torch.stack(predictions)


#sample code for training NHiTS on ETTm2

# if __name__ == "__main__":
#     model = NHiTS_Redesign(
#         3,
#         480,
#         96,
#         3,
#         [[128, 64, 32], [32, 16, 8], [4, 2, 1]],
#         "max",
#         128,
#         3,
#         [16, 64, 128],
#         [8, 32, 64],
#     )
#     cols = pd.read_csv("./data/ETTm2.csv").columns
#     for col in cols:
#         df = pd.read_csv("./data/ETTm2.csv")["MULL"]
#
#         dataset = NHiTS_Dataset(df, 480, 96)
#
#         dl = DataLoader(dataset, 128, shuffle=True)
#
#         model.fit(dl, nn.MSELoss(), 0.001, torch.optim.AdamW, 10)
