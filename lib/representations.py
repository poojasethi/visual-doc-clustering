import json
from collections import defaultdict
from enum import Enum
from functools import cached_property
from logging import getLogger
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
from attr import define, field
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

from lib.LayoutLM import LayoutLM
from lib.LayoutLMv2 import LayoutLMv2

from .path_utils import list_dirnames, only_file, walk

logger = getLogger(__name__)


class RepresentationType(str, Enum):
    RIVLET_COUNT = "rivlet_count"
    RIVLET_TFIDF = "rivlet_tfidf"
    LAYOUT_LM = "vanilla_lmv1"


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
    vectorized: Dict[str, np.array] = field(init=False)

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
    data_path: str, rep_type: str, models_dir: Optional[Path] = None
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
    elif rep_type == RepresentationType.LAYOUT_LM:
        data = prepare_representations_for_layout_lmv1(data)
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


def prepare_representations_for_layout_lmv1(
    data: CollectionRepresentations, model_path: Optional[Path] = None
) -> CollectionRepresentations:
    lm = LayoutLM()

    for documents in data.values():
        for doc, representation in documents.items():
            lm.process_json(representation.rivlet_path, "processed_word", "location", position_processing=True)
            lm.get_encodings()
            # TODO(pooja):  Hidden state  is a 512 x 768 vector. 512 is the length of the sequence and we need to average across this dimension. There are different things we can try here.
            hidden_state = lm.get_hidden_state(model_path)
            representation.vectorized[RepresentationType.LAYOUT_LM] = hidden_state["last_hidden_state"][0][0].numpy()
            documents[doc] = representation

    return data
