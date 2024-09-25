import os

import torch
from torch import nn

from image_encoder import CLIPVisionModel


class BLIP2(nn.Moudle):
    
    def init_tokenizer(self,):
        self.tokenizer = BertTokenizer.from_pretrained("klue/bert-base")
        self.tokenizer.add_special_tokens({"bos_token": "[DEC]"})

    def init_image_encoder(self, vision_model_name_or_path):
        self.image_encoder = CLIPVisionModel(vision_model_name_or_path)

    def init_llm():
        pass
