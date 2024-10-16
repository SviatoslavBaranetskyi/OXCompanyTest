import pandas as pd
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image

# Настройки
MODEL_PATH = 'anime_model.pth'
OUTPUT_CSV = 'clustered_labels.csv'


class MyModel(nn.Module):
    def __init__(self, num_classes):
        super(MyModel, self).__init__()
        self.resnet = models.resnet18(weights='IMAGENET1K_V1')
        self.resnet.fc = nn.Linear(self.resnet.fc.in_features, num_classes)

    def forward(self, x):
        return self.resnet(x)


labels_df = pd.read_csv(OUTPUT_CSV)
label_map = {idx: label for idx, label in enumerate(labels_df['cluster_label'].unique())}


transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


model = MyModel(num_classes=len(label_map))
model.load_state_dict(torch.load(MODEL_PATH, map_location=torch.device('cpu'), weights_only=True))
model.eval()


def predict(image_path, model, transform, label_map):
    image = Image.open(image_path).convert('RGB')
    image = transform(image).unsqueeze(0)

    with torch.no_grad():
        outputs = model(image)
        probabilities = nn.Softmax(dim=1)(outputs)
        confidence, predicted_class = torch.max(probabilities, 1)

    predicted_label = label_map[predicted_class.item()]
    confidence_score = confidence.item()

    return predicted_label, confidence_score
