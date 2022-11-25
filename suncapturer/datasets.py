import numpy as np
import torch
from torch.utils.data import Dataset


class SolarDataset(Dataset):
    def __init__(self, df, features):
        self.features = features
        self.X, self.y = self._build(df)

    def _build(self, df):
        start, end = df['fcst_time'].min(), df['fcst_time'].max()
        X, y = [], []
        for i in range(21):
            for day in range(start, end + 1):
                target_df = df[(df['id'] == i) & (df['fcst_time'] == day)]
                if (len(target_df) == 24) and (target_df['amount'].sum() != 0):
                    data = target_df[self.features].values
                    label = target_df['relative_amount'].values

                    X.append(data)
                    y.append(label)

        X, y = map(np.stack, [X, y])
        X = np.transpose(X, (0, 2, 1))
        X, y = map(lambda x: torch.from_numpy(x).float(), [X, y])

        return X, y

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

    def __len__(self):
        return len(self.y)
