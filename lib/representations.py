import json
import logging
from collections import defaultdict
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
from attr import define, field
from nptyping import NDArray
from sklearn.decomposition import PCA
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from tqdm import tqdm

from lib.layoutlm import LayoutLMPretrainedModel, LAYOUTLM_BASE, LAYOUTLM_LARGE, LAYOUTLM_V2
from lib.image_features import get_image_features

from .path_utils import list_dirnames, only_file, walk

logger = logging.getLogger(__name__)


class RepresentationType(str, Enum):
    RIVLET_COUNT = "rivlet_count"
    RIVLET_TFIDF = "rivlet_tfidf"
    RESNET = "resnet"
    ALEXNET = "alexnet"
    LAYOUTLM_BASE = "layoutlm_base"
    LAYOUTLM_LARGE = "layoutlm_large"
    LAYOUTLM_V2 = "layoutlm_v2"


layoutlm_rep_to_model_type = {
    RepresentationType.LAYOUTLM_BASE: LAYOUTLM_BASE,
    RepresentationType.LAYOUTLM_LARGE: LAYOUTLM_LARGE,
    RepresentationType.LAYOUTLM_V2: LAYOUTLM_V2,
}


class SquashStrategy(str, Enum):
    AVERAGE_ALL_WORDS = "average_all_words"
    AVERAGE_ALL_WORDS_MASK_PADS = "average_all_words_mask_pads"
    LAST_WORD = "last_word"
    PCA = "pca"


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
    data_path: str,
    rep_type: str,
    models_dir: Optional[Path] = None,
    squash_strategy: SquashStrategy = SquashStrategy.AVERAGE_ALL_WORDS_MASK_PADS,
    normalize_length: bool = False,
    exclude_length: bool = False,
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
        RepresentationType.LAYOUTLM_V2,
    ):
        data = prepare_representations_for_layout_lm(
            data,
            rep_type,
            squash_strategy=squash_strategy,
            normalize_length=False,
            exclude_length=exclude_length,
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
    data: CollectionRepresentations,
    rep_type: RepresentationType,
    squash_strategy: SquashStrategy,
    normalize_length: bool = False,
    exclude_length: bool = False,
) -> CollectionRepresentations:

    model_type = layoutlm_rep_to_model_type[rep_type]
    lm = LayoutLMPretrainedModel(model_type)

    for collection in data.values():
        for doc, representation in tqdm(collection.items()):
            hidden_states, attention_mask = lm.get_hidden_states(
                rivlets_path=representation.rivlet_path,
                image_path=representation.first_page_path,
            )
            sequence_length = np.sum(attention_mask)

            representation.vectorized[rep_type] = squash_hidden_states(
                hidden_states,
                attention_mask,
                squash_strategy,
                sequence_length=sequence_length,
                normalize_length=normalize_length,
                exclude_length=exclude_length,
            )
            collection[doc] = representation


def squash_hidden_states(
    hidden_states: NDArray,
    attention_mask: NDArray,
    squash_strategy: SquashStrategy,
    sequence_length: int,
    normalize_length: bool = False,
    exclude_length: bool = False,
) -> NDArray:
    """
    Squashes hidden_state matrix into a vector.
    TODO(pooja): Investigate ways hidden states are generally combined to form sentence vectors.
    """
    logger.info(f"Squashing hidden states using {squash_strategy}")
    if normalize_length:
        sequence_length /= 512

    # TODO(pooja): Add first (<CLS>) token

    if squash_strategy == SquashStrategy.AVERAGE_ALL_WORDS:
        return np.mean(hidden_states, axis=0)
    elif squash_strategy == SquashStrategy.AVERAGE_ALL_WORDS_MASK_PADS:
        non_pad_words = np.expand_dims(attention_mask, axis=1) * hidden_states
        average = np.mean(non_pad_words, axis=0)
        output = np.append(average, sequence_length) if not exclude_length else average
        return output
    elif squash_strategy == SquashStrategy.LAST_WORD:
        last_word_mask = np.zeros(attention_mask.shape[0])
        last_word_mask[sequence_length - 1] = 1
        last_word = np.sum(np.expand_dims(last_word_mask, axis=1) * hidden_states, axis=0)
        output = np.append(last_word, sequence_length) if not exclude_length else last_word
        return output
    elif squash_strategy == SquashStrategy.PCA:
        non_pad_words = np.expand_dims(attention_mask, axis=1) * hidden_states
        pca_hs = PCA(n_components=1)
        pca_output = pca_hs.fit_transform(non_pad_words.T)
        sequence_length = np.sum(attention_mask)
        output = np.append(pca_output, sequence_length) if not exclude_length else pca_output
        return output
    else:
        raise ValueError(f"Unknown squash strategy: {squash_strategy}")
