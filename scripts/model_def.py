import torch
import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights

class MultimodalPillClassifier(nn.Module):
    def __init__(self, text_feat_dim, num_classes):
        super().__init__()
        weights = ResNet18_Weights.DEFAULT
        self.cnn = resnet18(weights=weights)
        self.cnn.fc = nn.Identity()  # output size = 512

        self.text_fc = nn.Linear(text_feat_dim, 128)
        self.classifier = nn.Sequential(
            nn.Linear(512 + 128, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, num_classes)
        )

    def forward(self, image, text_features):
        img_feat = self.cnn(image)  # shape: (B, 512)
        text_feat = self.text_fc(text_features)  # shape: (B, 128)
        combined = torch.cat((img_feat, text_feat), dim=1)
        return self.classifier(combined)
