import glob
import json
import re
from pathlib import Path
from typing import Dict, List, Set, Tuple, Union

import numpy as np
import pandas as pd
import torch
from PIL import Image
from transformers import LayoutLMv2Model, LayoutLMv2Processor

from datasets import Array2D, Array3D, Array4D, ClassLabel, Dataset, Features, Sequence, Value, concatenate_datasets


class LayoutLMv2:
    """Get embeddings for LayoutLMv2"""

    def __init__(
        self,
        vocab_size=30522,
        hidden_size=768,
        num_hidden_layers=12,
        num_attention_heads=12,
        intermediate_size=3072,
        hidden_act="gelu",
        hidden_dropout_prob=0.1,
        attention_probs_dropout_prob=0.1,
        max_position_embeddings=512,
        type_vocab_size=2,
        initializer_range=0.02,
        layer_norm_eps=1e-12,
        pad_token_id=0,
        max_2d_position_embeddings=1024,
    ):

        # Initializing a LayoutLM configuration with default values
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.intermediate_size = intermediate_size
        self.hidden_act = hidden_act
        self.hidden_dropout_prob = hidden_dropout_prob
        self.attention_probs_dropout_prob = attention_probs_dropout_prob
        self.max_position_embeddings = max_position_embeddings
        self.type_vocab_size = type_vocab_size
        self.initializer_range = initializer_range
        self.layer_norm_eps = layer_norm_eps
        self.pad_token_id = pad_token_id
        self.max_2d_position_embeddings = max_2d_position_embeddings

        # Initializing a model from the configuration
        self.processor = LayoutLMv2Processor.from_pretrained("microsoft/layoutlmv2-base-uncased")
        self.model = LayoutLMv2Model.from_pretrained("microsoft/layoutlmv2-base-uncased")

        self.encoding = pd.DataFrame()

    def reset_encodings(self):
        self.encoding = pd.DataFrame()

    def __get_encodings(self, example):

        image = Image.open(example["image_path"]).convert("RGB")
        encoding = self.processor(image, padding="max_length", truncation=True)
        example["token_type_ids"] = encoding.token_type_ids[0]
        example["attention_mask"] = encoding.attention_mask[0]
        example["bbox"] = encoding.bbox[0]
        example["image"] = np.array(encoding.image[0])
        example["input_ids"] = encoding.input_ids[0]
        return example

    def __get_hidden_states(self, example, model=None):
        image = Image.open(example["image_path"]).convert("RGB")
        outputs = self.model(**self.processor(image, padding="max_length", truncation=True, return_tensors="pt"))
        example["last_hidden_state"] = outputs.last_hidden_state[0]
        return example

    def get_outputs(self, path, image_path=None, model=None, outpath=None, file_type="png"):

        # Import Files
        path = Path(path)

        if path.is_dir():
            files = path.rglob("*." + file_type)
        else:
            files = [path]

        # Select just one image to use -- need to think about whether we want to use more than 1 page
        # Should we concatenate the two hidden states?
        image_paths = (
            [image_path]
            if image_path
            else [
                re.match(r"(.+)/pages/0/(.*)", str(f)).group(0)
                for f in [*files]
                if re.match(r"(.+)/pages/0/(.*)", str(f)) is not None
            ]
        )

        # Testing a smaller number of images at the moment
        i = 0
        for ip in image_paths[:5]:
            self.encoding.at[i, "image_path"] = ip
            i += 1

        # Encode via mapping
        self.encoding = Dataset.from_pandas(self.encoding).remove_columns("__index_level_0__")

        features = Features(
            {
                # "image": Sequence(Sequence(Sequence(Sequence(Value(dtype="uint8"))))),
                "image": Array3D(dtype="int64", shape=(3, 224, 224)),
                "input_ids": Sequence(Value(dtype="int64")),
                "bbox": Array2D(dtype="int64", shape=(512, 4)),
                "attention_mask": Sequence(Value(dtype="int64")),
                "token_type_ids": Sequence(Value(dtype="int64")),
                "image_path": Value(dtype="string"),
            }
        )

        self.int_data = self.encoding.map(lambda example: self.__get_encodings(example), features=features)

        if model is None:
            model = self.model
        self.fin_data = self.int_data.map(lambda example: self.__get_hidden_states(example, model))

        self.fin_data.set_format(
            type="torch", columns=["image", "bbox", "attention_mask", "token_type_ids", "input_ids"]
        )

        if outpath is not None:
            self.fin_data.to_pandas().to_pickle(outpath)

        return self.fin_data


if __name__ == "__main__":

    directory = "/Users/bryanchia/Desktop/stanford/classes/cs/cs224n/project/data/files"

    i2 = LayoutLMv2()
    encodings = i2.get_outputs(directory)

    print(encodings)
