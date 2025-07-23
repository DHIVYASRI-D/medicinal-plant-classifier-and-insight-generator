import os
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from transformers import AutoImageProcessor
import torch

# Paths
DATASET_DIR = r"D:\leaf_dataset"
MODEL_NAME = "google/vit-base-patch16-224-in21k"
BATCH_SIZE = 8  # safe for CPU

# Load image processor
image_processor = AutoImageProcessor.from_pretrained(MODEL_NAME)

# Custom Dataset
class LeafDataset(Dataset):
    def __init__(self, root_dir, processor):
        self.root_dir = root_dir
        self.processor = processor
        self.image_paths = []
        self.labels = []
        self.class_to_idx = {}

        # List all images and labels
        for idx, class_name in enumerate(sorted(os.listdir(root_dir))):
            class_path = os.path.join(root_dir, class_name)
            if not os.path.isdir(class_path):
                continue
            self.class_to_idx[class_name] = idx
            for fname in os.listdir(class_path):
                if fname.lower().endswith((".jpg", ".jpeg", ".png")):
                    self.image_paths.append(os.path.join(class_path, fname))
                    self.labels.append(idx)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        label = self.labels[idx]

        image = Image.open(image_path).convert("RGB")
        encoding = self.processor(images=image, return_tensors="pt")

        # Remove batch dimension
        item = {key: val.squeeze(0) for key, val in encoding.items()}
        item["labels"] = torch.tensor(label)
        return item

# Dataloaders
def get_dataloaders(batch_size=BATCH_SIZE):
    train_ds = LeafDataset(os.path.join(DATASET_DIR, "train"), image_processor)
    val_ds   = LeafDataset(os.path.join(DATASET_DIR, "val"), image_processor)
    test_ds  = LeafDataset(os.path.join(DATASET_DIR, "test"), image_processor)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader   = DataLoader(val_ds, batch_size=batch_size)
    test_loader  = DataLoader(test_ds, batch_size=batch_size)

    return train_loader, val_loader, test_loader

if __name__ == "__main__":
    train_loader, val_loader, test_loader = get_dataloaders()
    print("Sample batch from training loader:")
    batch = next(iter(train_loader))
    print({k: v.shape for k, v in batch.items()})

print(train_loader.dataset.class_to_idx)
print(len(train_loader.dataset.class_to_idx))
