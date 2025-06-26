import torch
import timm
import numpy as np
from torchvision import transforms
from PIL import Image
from tqdm import tqdm


def get_vit_model(device='cpu'):
    """
    Loads pretrained ViT model.
    """
    model = timm.create_model('vit_base_patch16_224', pretrained=True)
    model.eval()
    model.to(device)
    return model


def get_preprocessing_transform():
    """
    Standard ViT image preprocessing.
    """
    return transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[
                             0.229, 0.224, 0.225]),
    ])


def extract_features(model, image_paths, device='cpu', batch_size=16, save_path=None):
    """
    Extracts features from images using ViT model, saves as .npy file if save_path given.
    """
    preprocess = get_preprocessing_transform()
    features = []

    for i in tqdm(range(0, len(image_paths), batch_size)):
        batch_paths = image_paths[i:i+batch_size]
        batch_imgs = []
        for path in batch_paths:
            img = Image.open(path).convert('RGB')
            batch_imgs.append(preprocess(img))
        batch_tensor = torch.stack(batch_imgs).to(device)
        with torch.no_grad():
            batch_feat = model.forward_features(batch_tensor)
        features.append(batch_feat.cpu().numpy())

    features = np.concatenate(features, axis=0)
    if save_path:
        np.save(save_path, features)
    return features
