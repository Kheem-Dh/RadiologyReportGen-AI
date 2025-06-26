from src.feature_extraction import get_vit_model, extract_features
from src.data_preprocessing import get_image_paths
import pandas as pd

# Paths
merged_csv = "data/merged_dataset.csv"
features_path = "data/chest_xray_features.npy"

df = pd.read_csv(merged_csv)
image_paths = get_image_paths(df)
model = get_vit_model(device='cuda')
features = extract_features(
    model, image_paths, device='cuda', batch_size=16, save_path=features_path)
print("Features extracted and saved.")
