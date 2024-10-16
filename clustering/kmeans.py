import os
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from torchvision import models, transforms
from PIL import Image
import torch

DATASET_PATH = 'data/images'
OUTPUT_CSV = 'clustered_labels.csv'
NUM_CLUSTERS = 1000

model = models.resnet50(weights='ResNet50_Weights.IMAGENET1K_V1')
model.eval()

preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


def get_features(image_path):
    image = Image.open(image_path).convert('RGB')
    image = preprocess(image).unsqueeze(0)
    with torch.no_grad():
        features = model(image)
    return features.numpy()


features_list = []
image_paths = []

for filename in os.listdir(DATASET_PATH):
    if filename.endswith('.jpg') or filename.endswith('.png'):
        image_path = os.path.join(DATASET_PATH, filename)
        features = get_features(image_path)
        features_list.append(features)
        image_paths.append(filename)

features_array = np.vstack(features_list)

scaler = StandardScaler()
features_scaled = scaler.fit_transform(features_array)

kmeans = KMeans(n_clusters=NUM_CLUSTERS, random_state=0)
labels = kmeans.fit_predict(features_scaled)

character_names = [f'Character_{i+1}' for i in range(NUM_CLUSTERS)]
label_names = [character_names[label] for label in labels]

results = pd.DataFrame({'image_name': image_paths, 'cluster_label': label_names})
results.to_csv(OUTPUT_CSV, index=False)

print(f'Clustering is complete! The results are saved in {OUTPUT_CSV}.')