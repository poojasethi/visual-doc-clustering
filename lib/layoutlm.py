import logging
from pathlib import Path
from typing import Optional, Tuple

import numpy.typing as npt
import tensorflow as tf
import torch
from PIL import Image
from transformers import (
    LayoutLMConfig,
    LayoutLMTokenizerFast,
    TFLayoutLMModel,
    LayoutLMv2Config,
    LayoutLMv2Model,
    LayoutLMv2Processor,
)

logger = logging.getLogger(__name__)

from lib.data_utils import encode_rivlets, get_words_and_bounding_boxes

# References: https://huggingface.co/microsoft/layoutlm-large-uncased
LAYOUTLM_BASE = "microsoft/layoutlm-base-uncased"
LAYOUTLM_LARGE = "microsoft/layoutlm-large-uncased"
LAYOUTLM_V2 = "microsoft/layoutlmv2-base-uncased"


DEFAULT_LAYOUTLM = LAYOUTLM_BASE


class LayoutLMPretrainedModel:
    def __init__(self, model_type: str) -> None:
        self.model_type = model_type

        if self.model_type in (LAYOUTLM_BASE, LAYOUTLM_LARGE):
            self.config = LayoutLMConfig.from_pretrained(self.model_type)
            self.tokenizer = LayoutLMTokenizerFast.from_pretrained(self.model_type)
            self.model = TFLayoutLMModel.from_pretrained(self.model_type)
        elif self.model_type in (LAYOUTLM_V2):
            self.config = LayoutLMv2Config.from_pretrained(self.model_type)
            self.processor = LayoutLMv2Processor.from_pretrained(self.model_type, revision="no_ocr")
            self.model = LayoutLMv2Model.from_pretrained(self.model_type)
        else:
            raise ValueError(f"Unrecognized model type: {self.model_type}")

        self.max_sequence_length = self.config.max_position_embeddings

    def get_hidden_states(
        self, rivlets_path: Path, image_path: Optional[Path] = None, verbose: bool = True
    ) -> Tuple[npt.NDArray, npt.NDArray]:

        token_embeddings, mask = None, None

        if self.model_type in [LAYOUTLM_BASE, LAYOUTLM_LARGE]:
            token_embeddings, mask = self._get_layoutlm_v1_hidden_states(rivlets_path, verbose=verbose)
        elif self.model_type in [LAYOUTLM_V2]:
            if not image_path:
                raise ValueError("Image path must be provided to use LayoutLMv2")
            token_embeddings, mask = self._get_layoutlm_v2_hidden_states(rivlets_path, image_path, verbose=verbose)

        assert token_embeddings is not None
        assert mask is not None

        if verbose:
            logger.info(f"Got token embeddings of shape: {token_embeddings.shape}")
            logger.info(f"Got mask of shape: {mask.shape}")

        # Write embeddings to path
        return token_embeddings, mask

    def _get_layoutlm_v1_hidden_states(
        self,
        rivlets_path: Path,
        verbose: bool = False,
    ) -> Tuple[npt.NDArray, npt.NDArray]:
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

        return embeddings, mask

    def _get_layoutlm_v2_hidden_states(
        self,
        rivlets_path: Path,
        image_path: Path,
        verbose: bool = False,
    ) -> Tuple[npt.NDArray, npt.NDArray]:
        image = Image.open(image_path).convert("RGB")
        words, word_bounding_boxes = get_words_and_bounding_boxes(rivlets_path)
        encoding = self.processor(image, words, boxes=word_bounding_boxes, return_tensors="pt")
        attention_mask = encoding["attention_mask"]

        outputs = self.model(**encoding)

        # Get the token embeddings from the last layer
        last_hidden_states = outputs.last_hidden_state

        # NOTE(pooja): The output returned by the model is of shape (batch_size, sequence_length, hidden_size).
        # Similarly, the attention mask is of shape (batch_size, sequence_length). Because batch_size == 1, we can
        # squeeze out the first dimension.
        embeddings = torch.squeeze(last_hidden_states)
        mask = torch.squeeze(attention_mask)

        assert (
            embeddings.shape[0] == mask.shape[0] == self.max_sequence_length
        ), f"Expected embeddings and mask to have length of {self.max_sequence_length}"

        embeddings = embeddings.numpy()
        mask = mask.numpy()

        return embeddings, mask
