import torch
from torch import nn

from transformers import CLIPVisionModel, CLIPImageProcessor, CLIPVisionConfig


class CLIPVisionTransformer(nn.Module):
    def __init__(self, vision_model_name_or_path):
        super().__init__()
        self.vision_model_name_or_path = vision_model_name_or_path 
        self.model = CLIPVisionModel.from_pretrained(self.vision_model_name_or_path)

    def forward(self, images):
        visual_features = self.model(images)
        return visual_features
