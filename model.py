import torch.nn as nn
from torchvision.models import efficientnet_b4, EfficientNet_B4_Weights


def build_efficientnet_b4(num_classes: int, dropout_rate: float = 0.4, pretrained: bool = True):
    weights = EfficientNet_B4_Weights.IMAGENET1K_V1 if pretrained else None
    model = efficientnet_b4(weights=weights)

    in_features = model.classifier[1].in_features
    # Bloğu doğrudan yeniden tanımlayarak çift dropout oluşmasını engelle
    model.classifier = nn.Sequential(
        nn.Dropout(p=dropout_rate),
        nn.Linear(in_features, num_classes)
    )
    return model