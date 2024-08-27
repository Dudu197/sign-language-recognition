from .base_model import BaseModel
from torch import nn
from torchvision.models import efficientnet_b6, EfficientNet_B6_Weights


class EfficientNetB6Model(BaseModel):
    name = "efficientnet_b6"
    model = None
    transforms = None
    image_size = (224, 224)

    def __init__(self, num_classes: int):
        super().__init__(num_classes)
        self.model = efficientnet_b6(weights=EfficientNet_B6_Weights.IMAGENET1K_V1)
        # self.model.head = nn.Sequential(
        #     nn.Linear(512, num_classes),
        #     # nn.Softmax(dim=1)
        # )

        num_ftrs = self.model.classifier[1].in_features

        # Add an extra dense layer
        self.model.classifier = nn.Sequential(
            nn.BatchNorm1d(num_ftrs),
            nn.Linear(num_ftrs, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, num_classes)
        )

    def get_model(self):
        return self.model

    def get_fc_layer(self):
        return self.model.classifier

    def get_transformers(self):
        return self.transforms
