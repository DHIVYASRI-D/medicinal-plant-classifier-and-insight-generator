ef get_dataloaders(batch_size=BATCH_SIZE):
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