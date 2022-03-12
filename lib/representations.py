import json
from collections import defaultdict
from enum import Enum
from functools import cached_property
from logging import getLogger
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch
from attr import define, field
from nptyping import NDArray
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from tqdm import tqdm

from lib.LayoutLM import LayoutLM
from lib.LayoutLMv2 import LayoutLMv2

from .path_utils import list_dirnames, only_file, walk

logger = getLogger(__name__)


class RepresentationType(str, Enum):
    RIVLET_COUNT = "rivlet_count"
    RIVLET_TFIDF = "rivlet_tfidf"
    VANILLA_LMV1 = "vanilla_lmv1"
    FINETUNED_RELATED_LMV1 = "finetuned_related_lmv1"
    FINETUNED_UNRELATED_LMV1 = "finetuned_unrelated_lmv1"
    VANILLA_LMV2 = "vanilla_lmv2"


class SquashStrategy(str, Enum):
    AVERAGE_ALL_WORDS = "average_all_words"
    AVERAGE_ALL_WORDS_MASK_PADS = "average_all_words_mask_pads"
    LAST_WORD = "last_word"


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
    vectorized: Dict[str, NDArray] = field(init=False)

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
    data_path: str, rep_type: str, models_dir: Optional[Path] = None, squash_strategy: Optional[SquashStrategy] = None
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
        RepresentationType.VANILLA_LMV1,
        RepresentationType.FINETUNED_RELATED_LMV1,
        RepresentationType.FINETUNED_UNRELATED_LMV1,
        RepresentationType.VANILLA_LMV2,
    ):
        model_path = (
            None
            if rep_type in (RepresentationType.VANILLA_LMV1, RepresentationType.VANILLA_LMV2)
            else models_dir / rep_type
        )
        data = prepare_representations_for_layout_lm(
            data, rep_type, model_path=model_path, squash_strategy=squash_strategy
        )
    else:
        raise ValueError(f"Unknown representation type: {rep_type}")

    assert data, "Collection representations cannot be None."

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
    data: CollectionRepresentations,
    rep_type: RepresentationType,
    model_path: Optional[Path] = None,
    squash_strategy: SquashStrategy = SquashStrategy.AVERAGE_ALL_WORDS,
) -> CollectionRepresentations:
    lm = None
    if rep_type in (
        RepresentationType.VANILLA_LMV1,
        RepresentationType.FINETUNED_RELATED_LMV1,
        RepresentationType.FINETUNED_UNRELATED_LMV1,
    ):
        lm = LayoutLM()
        for collection in data.values():
            for doc, representation in tqdm(collection.items()):
                # NOTE(pooja): The current implementation is a bit hacky in that the model predictions are obtained
                # one-at-a-time. In the future, obtaining model predictions should be fully vectorized.
                lm.process_json(representation.rivlet_path, "processed_word", "location", position_processing=True)
                lm.get_encodings()

                lm_data = lm.get_hidden_state(model_path=model_path)
                hidden_states = torch.stack(lm_data["last_hidden_state"][0]).numpy()
                attention_mask = lm_data["attention_mask"][0].numpy()
                sequence_length = np.sum(attention_mask)

                representation.vectorized[rep_type] = squash_hidden_states(
                    hidden_states, attention_mask, squash_strategy, sequence_length=sequence_length
                )
                collection[doc] = representation
                lm.reset_preprocessed_data()

    elif rep_type in (RepresentationType.VANILLA_LMV2):
        lm = LayoutLMv2()
        for collection in data.values():
            for doc, representation in tqdm(collection.items()):
                lm_data = lm.get_outputs(
                    str(representation.rivlet_path), image_path=str(representation.first_page_path)
                )
                hidden_states = np.array([np.array(x) for x in lm_data["last_hidden_state"][0]])
                attention_mask = lm_data["attention_mask"][0].numpy()
                sequence_length = np.sum(attention_mask)

                # NOTE(bryan): Low-resolution image feature map is 7 x 7. When flattened, one obtains 49 image tokens.
                attention_image = np.ones(49)
                attention_mask = np.concatenate((attention_mask, attention_image), axis=0)

                representation.vectorized[rep_type] = squash_hidden_states(
                    hidden_states, attention_mask, squash_strategy, sequence_length
                )
                collection[doc] = representation
                lm.reset_encodings()
    else:
        raise ValueError(f"Unsupported representation type: {rep_type}")
    return data


def squash_hidden_states(
    hidden_states: NDArray, attention_mask: NDArray, squash_strategy: SquashStrategy, sequence_length: int
) -> NDArray:
    """
    Squashes hidden_state matrix into a vector.
    TODO(pooja): Investigate ways hidden states are generally combined to form sentence vectors.
    """
    logger.info(f"Squashing hidden states using {squash_strategy}")

    if squash_strategy == SquashStrategy.AVERAGE_ALL_WORDS:
        return np.mean(hidden_states, axis=0)
    elif squash_strategy == SquashStrategy.AVERAGE_ALL_WORDS_MASK_PADS:
        non_pad_words = np.expand_dims(attention_mask, axis=1) * hidden_states
        average = np.mean(non_pad_words, axis=0)
        sequence_length = np.sum(attention_mask)
        output = np.append(average, sequence_length)
        return output
    elif squash_strategy == SquashStrategy.LAST_WORD:
        last_word_mask = np.zeros(attention_mask.shape[0])
        last_word_mask[sequence_length - 1] = 1
        last_word = np.sum(np.expand_dims(last_word_mask, axis=1) * hidden_states, axis=0)
        output = np.append(last_word, sequence_length)
        return output
    else:
        raise ValueError(f"Unknown squash strategy: {squash_strategy}")
