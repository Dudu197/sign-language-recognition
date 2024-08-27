from .base_model import BaseModel
from torch import nn
import timm


class MobilenetV5HybridMediumModel(BaseModel):
    name = "mobilenet_v4_hybrid_medium"
    model = None
    transforms = None
    image_size = (256, 256)

    def __init__(self, num_classes: int):
        super().__init__(num_classes)
        self.model = timm.create_model(
            'mobilenetv4_hybrid_medium.e500_r224_in1k',
            pretrained=True,
            # num_classes=num_classes,  # remove classifier nn.Linear
        )
        # self.model.classifier = nn.Sequential(
        #     nn.Linear(1280, num_classes),
        # )

        self.model.classifier = nn.Sequential(
            nn.BatchNorm1d(1280),
            nn.Linear(1280, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, num_classes)
        )

        data_config = timm.data.resolve_model_data_config(self.model)
        self.transforms = timm.data.create_transform(**data_config, is_training=False)

    def get_model(self):
        return self.model

    def get_fc_layer(self):
        return self.model.classifier

    def get_transformers(self):
        return self.transforms
