import logging
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import numpy.typing as npt
import tensorflow as tf
import torch
from PIL import Image
from transformers import (
    LayoutLMConfig,
    LayoutLMTokenizerFast,
    LayoutLMv2Config,
    LayoutLMv2Model,
    LayoutLMv2Processor,
    TFLayoutLMModel,
)

# from transformers.models.layoutlmv2.modeling_layoutlmv2 import LayoutLMv2Model  # for debugging

logger = logging.getLogger(__name__)

from lib.data_utils import encode_rivlets, get_words_and_bounding_boxes

# References: https://huggingface.co/microsoft/layoutlm-large-uncased
LAYOUTLM_BASE = "microsoft/layoutlm-base-uncased"
LAYOUTLM_LARGE = "microsoft/layoutlm-large-uncased"
LAYOUTLM_V2_BASE = "microsoft/layoutlmv2-base-uncased"
LAYOUTLM_V2_LARGE = "microsoft/layoutlmv2-large-uncased"

DEFAULT_LAYOUTLM = LAYOUTLM_BASE


class LayoutLMPretrainedModel:
    def __init__(self, model_type: str) -> None:
        self.model_type = model_type

        if self.model_type in (LAYOUTLM_BASE, LAYOUTLM_LARGE):
            self.config = LayoutLMConfig.from_pretrained(self.model_type)
            self.tokenizer = LayoutLMTokenizerFast.from_pretrained(self.model_type)
            self.model = TFLayoutLMModel.from_pretrained(self.model_type)
        elif self.model_type in (LAYOUTLM_V2_BASE, LAYOUTLM_V2_LARGE):
            self.config = LayoutLMv2Config.from_pretrained(self.model_type)
            self.processor = LayoutLMv2Processor.from_pretrained(self.model_type, revision="no_ocr")
            self.model = LayoutLMv2Model.from_pretrained(self.model_type)
            self.image_feature_pool_shape = self.config.image_feature_pool_shape
        else:
            raise ValueError(f"Unrecognized model type: {self.model_type}")

        self.max_sequence_length = self.config.max_position_embeddings

    def get_hidden_states(
        self, rivlets_path: Path, image_path: Optional[Path] = None, verbose: bool = True
    ) -> Tuple[npt.NDArray, npt.NDArray, int, int]:

        token_embeddings, mask = None, None

        if self.model_type in [LAYOUTLM_BASE, LAYOUTLM_LARGE]:
            token_embeddings, mask, sequence_length, image_length = self._get_layoutlm_v1_hidden_states(
                rivlets_path, verbose=verbose
            )
        elif self.model_type in [LAYOUTLM_V2_BASE, LAYOUTLM_V2_LARGE]:
            if not image_path:
                raise ValueError("Image path must be provided to use LayoutLMv2")
            token_embeddings, mask, sequence_length, image_length = self._get_layoutlm_v2_hidden_states(
                rivlets_path, image_path, verbose=verbose
            )

        assert token_embeddings is not None
        assert mask is not None

        if verbose:
            logger.info(f"Got token embeddings of shape: {token_embeddings.shape}")
            logger.info(f"Got mask of shape: {mask.shape}")

        # Write embeddings to path
        return token_embeddings, mask, sequence_length, image_length

    def _get_layoutlm_v1_hidden_states(
        self,
        rivlets_path: Path,
        verbose: bool = False,
    ) -> Tuple[npt.NDArray, npt.NDArray, int, int]:
        encoding = encode_rivlets(self.tokenizer, rivlets_path, self.max_sequence_length, verbose=verbose)

        input_ids = encoding["input_ids"]
        attention_mask = encoding["attention_mask"]
        token_type_ids = encoding["token_type_ids"]
        bbox = encoding["bbox"]

        outputs = self.model(
            input_ids=input_ids,
            bbox=bbox,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
        )

        # Get the token embeddings from the last layer
        last_hidden_states = outputs.last_hidden_state

        # NOTE(pooja): The output returned by the model is of shape (batch_size, sequence_length, hidden_size).
        # Similarly, the attention mask is of shape (batch_size, sequence_length). Because batch_size == 1, we can
        # squeeze out the first dimension.
        embeddings = tf.squeeze(last_hidden_states)
        mask = tf.squeeze(attention_mask)

        assert (
            embeddings.shape[0] == mask.shape[0] == self.max_sequence_length
        ), f"Expected embeddings and mask to have length of {self.max_sequence_length}"

        embeddings = embeddings.numpy()
        mask = mask.numpy()

        sequence_length = np.sum(attention_mask)
        image_length = 0  # LayoutLM doesn't use the image, so the image_length is 0
        return embeddings, mask, sequence_length, image_length

    def _get_layoutlm_v2_hidden_states(
        self,
        rivlets_path: Path,
        image_path: Path,
        verbose: bool = False,
    ) -> Tuple[npt.NDArray, npt.NDArray, int, int]:
        image = Image.open(image_path).convert("RGB")
        words, word_bounding_boxes = get_words_and_bounding_boxes(rivlets_path)
        encoding = self.processor(
            image,
            words,
            boxes=word_bounding_boxes,
            padding="max_length",
            truncation=True,
            max_length=self.max_sequence_length,
            return_tensors="pt",
        )
        attention_mask = encoding["attention_mask"]

        # Get the token embeddings from the last layer
        with torch.no_grad():
            outputs = self.model(**encoding)

            last_hidden_states = outputs.last_hidden_state

            # NOTE(pooja): The output returned by the model is of shape (batch_size, sequence_length, hidden_size).
            # Similarly, the attention mask is of shape (batch_size, sequence_length). Because batch_size == 1, we can
            # squeeze out the first dimension.
            embeddings = torch.squeeze(last_hidden_states)

            attention_mask = torch.squeeze(attention_mask)
            sequence_length = torch.sum(attention_mask)
            image_feature_length = self.image_feature_pool_shape[0] * self.image_feature_pool_shape[1]

            visual_attention_mask = torch.ones(image_feature_length)
            final_attention_mask = torch.cat([attention_mask, visual_attention_mask], dim=0)

            # LayoutLMv2 concatenates the text features with 7 x 7 image features.
            # "This means that the last hidden states of the model will have a length of 512 + 49 = 561,
            # if you pad the text tokens up to the max length."
            # Reference: https://huggingface.co/docs/transformers/v4.19.2/en/model_doc/layoutlmv2#overview
            expected_hidden_state_length = self.max_sequence_length + image_feature_length

            assert (
                embeddings.shape[0] == expected_hidden_state_length
            ), f"Expected embeddings to have length of {expected_hidden_state_length}"

            assert (
                final_attention_mask.shape[0] == expected_hidden_state_length
            ), f"Expected mask to have length of {expected_hidden_state_length}"

            embeddings = embeddings.numpy()
            mask = final_attention_mask.numpy()

            return embeddings, mask, sequence_length, image_feature_length
