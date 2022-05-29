import json
import logging
from collections import Counter, OrderedDict
from transformers.utils.dummy_tokenizers_objects import LayoutLMTokenizerFast
from pathlib import Path
from typing import Any, List, Tuple, Optional

import tensorflow as tf
import numpy as np

from transformers.utils.dummy_tokenizers_objects import LayoutLMTokenizerFast

logger = logging.getLogger(__name__)

# We assign fake bounding boxes to the <CLS>, <SEP>, and <PAD> special tokens.
CLS_BOX = [0, 0, 0, 0]
SEP_BOX = [1000, 1000, 1000, 1000]
PAD_BOX = [1000, 0, 1000, 0]


def encode_rivlets(
    tokenizer: LayoutLMTokenizerFast, rivlets_dir: Path, max_sequence_length: int = 512, verbose: bool = False
):
    words, word_bounding_boxes = get_words_and_bounding_boxes(rivlets_dir, verbose)
    encoding = tokenizer(
        words,
        max_length=max_sequence_length,
        padding="max_length",
        truncation=True,
        is_split_into_words=True,
        return_tensors="tf",
    )
    if verbose:
        logger.info(f"Got tokens:\n{encoding.tokens()}")

    word_bounding_boxes = np.array(word_bounding_boxes)

    # For each word in the original input sequence, count how many tokens were created.
    word_idx_to_token_counts = OrderedDict(Counter(encoding.word_ids()))

    # Special tokens are mapped to "None". We can ignore them.
    # Ref: https://github.com/huggingface/transformers/blob/v4.18.0/src/transformers/tokenization_utils_base.py#L338
    if None in word_idx_to_token_counts:
        word_idx_to_token_counts.pop(None)

    if words:
        word_indices = list(word_idx_to_token_counts.keys())
        word_to_token_counts = list(word_idx_to_token_counts.values())
        token_bounding_boxes = word_bounding_boxes[word_indices]
        token_bounding_boxes_repeated = np.repeat(token_bounding_boxes, word_to_token_counts, axis=0)
    else:
        # Handle case where there are no words (empty document)
        token_bounding_boxes_repeated = np.array([])

    truncated_token_boxes = truncate_and_pad_sequence(
        token_bounding_boxes_repeated.tolist(),
        max_sequence_length,
        pad_value=PAD_BOX,
        start_value=CLS_BOX,
        end_value=SEP_BOX,
    )

    encoding["bbox"] = tf.convert_to_tensor([truncated_token_boxes])

    return encoding


def get_words_and_bounding_boxes(rivlets_path: Path, verbose: bool = True) -> Tuple[List[str], List[List[int]]]:
    words, word_bounding_boxes = load_rivlets(rivlets_path)

    word_bounding_boxes = np.array(word_bounding_boxes)

    # Mask out a bounding box (and its corresponding word) if any of its four coordinates are not between [0, 1].
    bbox_mask = ((word_bounding_boxes >= 0) & (word_bounding_boxes <= 1)).all(axis=1)

    if verbose:
        bbox_mask_inverse = ~bbox_mask

        remove_word_indices = np.flatnonzero(bbox_mask_inverse)
        removed_words = [words[i] for i in remove_word_indices]

        removed_boxes = word_bounding_boxes[bbox_mask_inverse].tolist()
        logger.info(f"Filtering out the following word and bbox pairs:\n{(list(zip(removed_words, removed_boxes)))}")

    word_indices = np.flatnonzero(bbox_mask)
    words = [words[i] for i in word_indices]

    word_bounding_boxes = word_bounding_boxes[bbox_mask]
    word_bounding_boxes_scaled = np.around(word_bounding_boxes * 1000.0).astype(int).clip(min=0).tolist()

    return words, word_bounding_boxes_scaled


def load_rivlets(rivlets_path: Path) -> Tuple[List[str], List[List[int]]]:
    """Gets all the rivlets and bounding boxes from the rivlets file. Does not do any filtering."""

    words = []
    bounding_boxes = []

    with open(rivlets_path, "r") as rivlets_fh:
        rivlets = json.load(rivlets_fh)

        for rivlet in rivlets:
            words.append(rivlet["word"])
            location = rivlet["location"]

            x0 = location["left"]
            y0 = location["top"]
            x1 = location["left"] + location["width"]
            y1 = location["top"] + location["height"]

            bounding_boxes.append([x0, y0, x1, y1])

    return words, bounding_boxes


def truncate_and_pad_sequence(
    sequence: List[Any],
    length: int,
    pad_value: Optional[Any] = None,
    start_value: Optional[Any] = None,
    end_value: Optional[Any] = None,
) -> List[Any]:
    additional_offset = 0

    if start_value:
        additional_offset += 1
    if end_value:
        additional_offset += 1

    # Account for the special start and end tokens.
    sequence = sequence[: length - additional_offset]

    # Likewise, if the sequence is too short, add "fake" values.
    # This is to account for the added <PAD> tokens.
    if len(sequence) < length - additional_offset:
        padding_length = (length - additional_offset) - len(sequence)
        sequence.extend([pad_value] * padding_length)

    if start_value:
        sequence = [start_value] + sequence
    if end_value:
        sequence = sequence + [end_value]

    assert len(sequence) == length

    return sequence
