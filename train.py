from dataset import ChatDataset
from torch.utils.data import DataLoader

from test import X_train, y_train

dataset = ChatDataset(X_train, y_train)

train_loader = DataLoader(
    dataset,
    batch_size=8,
    shuffle=True
)