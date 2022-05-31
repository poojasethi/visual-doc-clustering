import json
import logging
from collections import defaultdict
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import numpy.typing as npt
import torch
from attr import define, field
from sklearn.decomposition import PCA
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from tqdm import tqdm

from lib.image_features import get_image_features
from lib.layoutlm import LAYOUTLM_BASE, LAYOUTLM_LARGE, LAYOUTLM_V2_BASE, LAYOUTLM_V2_LARGE, LayoutLMPretrainedModel

from .path_utils import list_dirnames, only_file, walk

logger = logging.getLogger(__name__)


class RepresentationType(str, Enum):
    RIVLET_COUNT = "rivlet_count"
    RIVLET_TFIDF = "rivlet_tfidf"
    RESNET = "resnet"
    ALEXNET = "alexnet"
    LAYOUTLM_BASE = "layoutlm_base"
    LAYOUTLM_LARGE = "layoutlm_large"
    LAYOUTLM_V2_BASE = "layoutlm_v2_base"
    LAYOUTLM_V2_LARGE = "layoutlm_v2_large"


layoutlm_rep_to_model_type = {
    RepresentationType.LAYOUTLM_BASE: LAYOUTLM_BASE,
    RepresentationType.LAYOUTLM_LARGE: LAYOUTLM_LARGE,
    RepresentationType.LAYOUTLM_V2_BASE: LAYOUTLM_V2_BASE,
    RepresentationType.LAYOUTLM_V2_LARGE: LAYOUTLM_V2_LARGE,
}


class SquashStrategy(str, Enum):
    CLS_TOKEN = "cls_token"
    SEP_TOKEN = "sep_token"
    IMAGE_TOKENS = "image_tokens"
    AVERAGE_ALL_TOKENS = "average_all_tokens"


@define
class DocumentRepresentation:
    rivlets: List[Dict[str, Any]]

    # Path to first (0th) page of the document. Useful for displaying previews.
    first_page_path: Path

    # Pah to the original rivlets.json file for this document.
    rivlet_path: Path

    # All of the rivlets joined together into one "paragraph".
    rivlet_stream: str = field(init=False)

    # Mapping of representation type to vector. A document can have many possible representations.
    vectorized: Dict[str, npt.NDArray] = field(init=False)

    # Mapping of representation type to cluster assignment.
    cluster: Dict[str, int] = field(init=False)

    def __attrs_post_init__(self):
        self.rivlet_stream = " ".join([rivlet["processed_word"] for rivlet in self.rivlets])
        self.vectorized = {}
        self.cluster = {}


# TODO(pooja): The Document and DocumentRepresentation object should be combined into a single object.
@define
class Document:
    id: str
    representation: DocumentRepresentation

    def __hash__(self):
        return hash(self.id)

    def __eq__(self, other):
        return self.id == other.id


# TODO(pooja): It may  be possible to combine this data structure with Document.
# Instead of maintaining pointers of collection to documents, we could have each
# Document maintain a set of the collections it is a member of. See D12061 for
# further context.
#
# Mapping of collection to documents (files) it contains.
# In turn, each document is mapped to its representation.
CollectionRepresentations = Dict[str, Dict[str, DocumentRepresentation]]


def prepare_representations(
    data_path: str,
    rep_type: str,
    squash_strategy: SquashStrategy = SquashStrategy.AVERAGE_ALL_TOKENS,
) -> CollectionRepresentations:
    """
    Returns a mapping of collection to document (file) to vectorized representation.
    """
    path = Path(data_path)
    data = defaultdict(dict)
    for collection in list_dirnames(path):
        rivlets_dir = path / collection / "base" / "rivlets"
        try:
            for file_hash_prefix in list_dirnames(rivlets_dir):
                for rivlets_file in walk(rivlets_dir / file_hash_prefix):
                    file_id = rivlets_file.stem
                    try:
                        first_page_path = (
                            path / collection / "base" / "files" / file_hash_prefix / file_id / "pages" / "0"
                        )
                        first_page_file = only_file(first_page_path)
                    except Exception:
                        first_page_path = (
                            path / collection / "base" / "files" / file_hash_prefix / file_id / "normalized"
                        )
                        first_page_file = only_file(first_page_path)

                    assert first_page_file, f"Did not find first page under {first_page_path}"

                    with open(rivlets_file) as fh:
                        rivlets = json.loads(fh.read())
                        data[collection][file_id] = DocumentRepresentation(
                            rivlets=rivlets,
                            rivlet_path=rivlets_file,
                            first_page_path=first_page_file,
                        )
        except Exception as e:
            logger.warning(e)
            logger.warning(f"Failed to prepare document representations for {collection}. Skipping...")
            continue

    if rep_type == RepresentationType.RIVLET_COUNT:
        data = prepare_representations_for_rivlet_count(data)
    elif rep_type == RepresentationType.RIVLET_TFIDF:
        data = prepare_representations_for_rivlet_tfidf(data)
    elif rep_type in (
        RepresentationType.LAYOUTLM_BASE,
        RepresentationType.LAYOUTLM_LARGE,
        RepresentationType.LAYOUTLM_V2_BASE,
        RepresentationType.LAYOUTLM_V2_LARGE,
    ):
        data = prepare_representations_for_layout_lm(
            data,
            rep_type,
            squash_strategy=squash_strategy,
        )
    elif rep_type in (RepresentationType.RESNET, RepresentationType.ALEXNET):
        data = prepare_representations_for_image(data, rep_type)
    else:
        raise ValueError(f"Unknown representation type: {rep_type}")

    assert data, "Collection representations cannot be None."

    return data


def prepare_representations_for_image(
    data: CollectionRepresentations, rep_type: RepresentationType
) -> CollectionRepresentations:
    for documents in data.values():
        for doc, representation in documents.items():
            # Transform can take a list of documents, so this could be sped up by only calling it once.
            vector = get_image_features(representation.first_page_path, model=rep_type)
            representation.vectorized[rep_type] = vector.numpy()
            documents[doc] = representation

    return data


def prepare_representations_for_rivlet_count(data: CollectionRepresentations) -> CollectionRepresentations:
    vectorizer = CountVectorizer()
    corpus = [representation.rivlet_stream for documents in data.values() for representation in documents.values()]
    vectorizer.fit(corpus)

    for documents in data.values():
        for doc, representation in documents.items():
            # Transform can take a list of documents, so this could be sped up by only calling it once.
            vector = vectorizer.transform([representation.rivlet_stream])
            representation.vectorized[RepresentationType.RIVLET_COUNT] = vector.toarray()[0]
            documents[doc] = representation

    return data


def prepare_representations_for_rivlet_tfidf(data: CollectionRepresentations) -> CollectionRepresentations:
    vectorizer = TfidfVectorizer()
    corpus = [representation.rivlet_stream for documents in data.values() for representation in documents.values()]
    vectorizer.fit(corpus)

    for documents in data.values():
        for doc, representation in documents.items():
            # Transform can take a list of documents, so this could be sped up by only calling it once.
            vector = vectorizer.transform([representation.rivlet_stream])
            representation.vectorized[RepresentationType.RIVLET_TFIDF] = vector.toarray()[0]
            documents[doc] = representation

    return data


def prepare_representations_for_layout_lm(
    data: CollectionRepresentations, rep_type: RepresentationType, squash_strategy: SquashStrategy
) -> CollectionRepresentations:

    model_type = layoutlm_rep_to_model_type[rep_type]
    lm = LayoutLMPretrainedModel(model_type)

    for collection in data.values():
        for doc, representation in tqdm(collection.items()):
            hidden_states, attention_mask, sequence_length, image_length = lm.get_hidden_states(
                rivlets_path=representation.rivlet_path,
                image_path=representation.first_page_path,
            )

            hidden_state_length = hidden_states.shape[0]
            expected_mask = np.zeros(hidden_state_length)
            expected_mask[:sequence_length] = 1
            if image_length > 0:
                expected_mask[-image_length:] = 1

            assert np.all(
                attention_mask == expected_mask
            ), f"Given attention mask doesn't match expected mask for sequence length {sequence_length} and image feature length {image_length}"

            representation.vectorized[rep_type] = squash_hidden_states(
                hidden_states,
                attention_mask,
                squash_strategy,
                sequence_length,
                image_length,
            )
            collection[doc] = representation
    return data


def squash_hidden_states(
    hidden_states: npt.NDArray,
    attention_mask: npt.NDArray,
    squash_strategy: SquashStrategy,
    sequence_length: int,
    image_length: int,
    append_length: bool = False,
) -> npt.NDArray:
    """
    Squashes hidden_state matrix into a vector.
    TODO(pooja): Investigate ways hidden states are generally combined to form sentence vectors.
    """
    logger.info(f"Squashing hidden states using {squash_strategy}")
    # TODO(pooja): Add strategy for selecting first (<CLS>) token
    if squash_strategy == SquashStrategy.CLS_TOKEN:
        cls_token = hidden_states[0]
        assert cls_token.shape[0] == hidden_states.shape[1]
        output = cls_token
    elif squash_strategy == SquashStrategy.SEP_TOKEN:
        last_token = hidden_states[sequence_length - 1]
        assert last_token.shape[0] == hidden_states.shape[1]
        output = last_token
    elif squash_strategy == SquashStrategy.IMAGE_TOKENS:
        assert image_length > 0, "Must have non-zero image token length to use image_token squash strategy"
        image_tokens = hidden_states[-image_length:]
        assert image_tokens.shape[0] == image_length
        average = np.mean(image_tokens, axis=0)
        output = average
    elif squash_strategy == SquashStrategy.AVERAGE_ALL_TOKENS:
        if sequence_length > 2:
            attention_mask[0] = 0  # Remove the [CLS] token
            attention_mask[sequence_length - 1] = 0  # Remove the [SEP] token
            tokens = hidden_states[attention_mask.astype(bool)]
            tokens = tokens
            assert tokens.shape[0] == sequence_length + image_length - 2
            average = np.mean(tokens, axis=0)
            output = average
        else:
            # Use the <CLS> token if the document is empty.
            output = hidden_states[0]
    else:
        raise ValueError(f"Unknown squash strategy: {squash_strategy}")

    if append_length:
        output = np.append(output, sequence_length)

    return output
