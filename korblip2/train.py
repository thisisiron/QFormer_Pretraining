
import os
from dataclasses import dataclass, field
from typing import Optional

import transformers
from transformers import (
    AutoConfig,
    AutoImageProcessor,
    AutoModel,
    AutoTokenizer,
    HfArgumentParser,
    Trainer,
    TrainingArguments,
    set_seed,
)

from transformers.models.blip_2.modeling_blip_2 import Blip2Model


@dataclass
class ModelArguments:
    model_name_or_path: str = field(
        default="Salesforce/blip2-opt-2.7b",
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"},
    )
    config_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained config name or path if not the same as model_name"}
    )
    tokenizer_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
    )


@dataclass
class DataTrainingArguments:
    pass


@dataclass
class TrainingArguments(TrainingArguments):
    pass



def main():
    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    config = AutoConfig.from_pretrained(model_args.model_name_or_path,)
    model = Blip2Model.from_pretrained(
            model_args.model_name_or_path,
            config=config,
    )



if __name__ == "__main__":
    main()
