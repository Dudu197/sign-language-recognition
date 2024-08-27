from .base_model import BaseModel
from torch import nn
from torchvision.models import vit_l_16, ViT_L_16_Weights


class VitL16Model(BaseModel):
    name = "vit_l_16"
    model = None
    transforms = None
    image_size = (512, 512)

    def __init__(self, num_classes: int):
        super().__init__(num_classes)
        self.model = vit_l_16(weights=ViT_L_16_Weights.IMAGENET1K_SWAG_E2E_V1)
        # self.model.head = nn.Sequential(
        #     nn.Linear(512, num_classes),
        #     # nn.Softmax(dim=1)
        # )

        num_ftrs = self.model.heads.head.in_features

        # Add an extra dense layer
        self.model.heads = nn.Sequential(
            nn.BatchNorm1d(num_ftrs),
            nn.Linear(num_ftrs, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, num_classes)
        )

    def get_model(self):
        return self.model

    def get_fc_layer(self):
        return self.model.heads

    def get_transformers(self):
        return self.transforms
