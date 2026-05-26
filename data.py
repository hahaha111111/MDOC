from pathlib import Path

import pandas as pd
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms


class OneClassImageDataset(Dataset):
    def __init__(
        self,
        data_root,
        metadata,
        split,
        target_label,
        image_size=128,
        channels=1,
        path_column="path",
        label_column="label",
        split_column="split",
        train=False,
    ):
        self.data_root = Path(data_root)
        self.target_label = str(target_label)
        self.path_column = path_column
        self.label_column = label_column
        self.channels = channels
        frame = pd.read_csv(metadata)
        for column in [path_column, label_column, split_column]:
            if column not in frame.columns:
                raise ValueError(f"Missing column: {column}")
        frame = frame[frame[split_column].astype(str).str.lower() == split.lower()].copy()
        if train:
            frame = frame[frame[label_column].astype(str) == self.target_label].copy()
        if frame.empty:
            raise ValueError("No samples found for the requested split")
        self.frame = frame.reset_index(drop=True)
        steps = [transforms.Resize((image_size, image_size)), transforms.ToTensor()]
        self.transform = transforms.Compose(steps)

    def __len__(self):
        return len(self.frame)

    def __getitem__(self, index):
        row = self.frame.iloc[index]
        image_path = Path(str(row[self.path_column]))
        if not image_path.is_absolute():
            image_path = self.data_root / image_path
        mode = "L" if self.channels == 1 else "RGB"
        image = Image.open(image_path).convert(mode)
        label = 0 if str(row[self.label_column]) == self.target_label else 1
        return self.transform(image), label


def create_loaders(
    data_root,
    metadata,
    target_label,
    image_size=128,
    channels=1,
    batch_size=64,
    workers=4,
    path_column="path",
    label_column="label",
    split_column="split",
):
    train_set = OneClassImageDataset(
        data_root=data_root,
        metadata=metadata,
        split="train",
        target_label=target_label,
        image_size=image_size,
        channels=channels,
        path_column=path_column,
        label_column=label_column,
        split_column=split_column,
        train=True,
    )
    test_set = OneClassImageDataset(
        data_root=data_root,
        metadata=metadata,
        split="test",
        target_label=target_label,
        image_size=image_size,
        channels=channels,
        path_column=path_column,
        label_column=label_column,
        split_column=split_column,
        train=False,
    )
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=workers, pin_memory=True)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=workers, pin_memory=True)
    return train_loader, test_loader
