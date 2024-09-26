import torch
from torch import nn

from transformers import CLIPVisionModel, CLIPImageProcessor, CLIPVisionConfig


class CLIPVisionTransformer(nn.Module):
    def __init__(self, vision_model_name_or_path):
        super().__init__()
        self.vision_model_name_or_path = vision_model_name_or_path 
        self.model = CLIPVisionModel.from_pretrained(self.vision_model_name_or_path)
        # self.config = CLIPVisionConfig.from_pretrained(self.vision_model_name_or_path)

    def forward(self, images):
        visual_features = self.model(images)
        return visual_features

    @property
    def config(self):
        return self.model.config

    @property
    def hidden_size(self):
        return self.config.hidden_size

    @property
    def num_patches(self):
        return (self.config.image_size // self.config.patch_size) ** 2

    @property
    def dtype(self):
        return self.model.dtype

    @property
    def device(self):
        return self.model.device
