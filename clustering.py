"""
This script clusters related documents together. Specifically, it does the following:
1. For a given set of collections, represents each document within them as a vector.
2. Applies a clustering algorithm over the vectorized document representations. The number of clusters are assumed to
   be unknown in advance.
3. Computes metrics and visualizations of clustering performance.

Examples:
python -m clustering.py -p <dataset_path>

"""

import argparse
import logging
import pprint
import statistics
from collections import Counter, defaultdict
from enum import Enum
from pathlib import Path
from typing import List, Optional

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from joblib import dump, load
from scipy.optimize import linear_sum_assignment
from sklearn import preprocessing
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.metrics import calinski_harabasz_score, classification_report, confusion_matrix, silhouette_score
from sklearn.mixture import BayesianGaussianMixture

from lib.path_utils import existing_directory
from lib.plot_utils import display_scatterplot
from lib.representations import CollectionRepresentations, RepresentationType, prepare_representations

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ClusteringAlgorithm(str, Enum):
    BAYESIAN_GAUSSIAN_MIXTURE = "bayesian_gaussian_mixture"


class ClusteringParameters:
    RANDOM_SEED: int = 42
    N_COMPONENTS: int = 10


def main(args: argparse.Namespace):
    logger.info("Running document clustering")

    # Vectorize the data, but before, check if it's already been vectorized and try loading it.
    prepared_data_path = args.output_path / "prepared_data.joblib"
    if prepared_data_path.exists():
        logger.info(f"Loading document representations from {prepared_data_path}")
        data = load(prepared_data_path)
    else:
        logger.info(f"Preparing document representations...")
        data = prepare_representations(
            args.data_path,
            args.representation,
            models_dir=args.models_path,
            squash_strategy=args.squash_strategy,
            normalize_length=args.normalize_length,
        )
        try:
            dump(data, prepared_data_path)
        except Exception as e:
            logger.warning(f"Failed to save embeddings to {prepared_data_path}")
            logger.warning(e)

    # Run clustering algorithm
    data = apply_clustering(data, args.representation, args.num_clusters, args.embedding_size)

    # Visualize the clusters and log metrics.
    plot_data_and_metrics(data, args.representation, args.debug, args.output_path)


def apply_clustering(
    data: CollectionRepresentations,
    rep_type: str,
    num_clusters: int,
    max_embedding_size: int,
) -> CollectionRepresentations:
    corpus_vectorized = np.array(
        [
            representation.vectorized[rep_type]
            for _, documents in data.items()
            for _, representation in documents.items()
        ]
    )

    embedding_size = len(corpus_vectorized[0])
    model = None  # Model to use for dimensionality reduction, if applicable.

    if embedding_size > max_embedding_size:
        Model = TruncatedSVD if rep_type in (RepresentationType.RIVLET_COUNT, RepresentationType.RIVLET_TFIDF) else PCA
        logger.info(f"Applying dimensionality reduction using {Model}")

        # n_components in PCA must be between [0,  min(n_samples, n_features)]
        # Refs:
        # https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.PCA.html#sklearn.decomposition.PCA
        embedding_size = min(len(corpus_vectorized), max_embedding_size) if Model == PCA else max_embedding_size
        logger.info(f"Set embedding size to {embedding_size}")

        model = Model(n_components=embedding_size)
        corpus_vectorized = model.fit_transform(corpus_vectorized)

    # TODO(pooja): Also set `weight_concentration_prior`, or use AdaptiveGMM.
    gmm = BayesianGaussianMixture(n_components=num_clusters, random_state=ClusteringParameters.RANDOM_SEED).fit(
        corpus_vectorized
    )

    for _, documents in data.items():
        for doc, representation in documents.items():
            # Predict can take a list of documents, so this could be sped up by only calling it once.
            # Alternatively, 'fit_predict' could be used here.
            vector = representation.vectorized[rep_type]
            cluster = gmm.predict(model.transform([vector]) if model else [vector])
            representation.cluster[rep_type] = cluster[0]
            documents[doc] = representation

    return data


def plot_data_and_metrics(
    data: CollectionRepresentations, rep_type: str, debug: bool = False, output_path: Optional[Path] = None
) -> None:
    corpus = [
        (
            representation.vectorized[rep_type],
            collection,
            representation.cluster[rep_type],
            representation.first_page_path,
        )
        for collection, documents in data.items()
        for _, representation in documents.items()
    ]

    corpus_vectorized, corpus_collections, corpus_clusters, first_pages = map(list, zip(*corpus))
    display_scatterplot(
        corpus_vectorized,
        corpus_collections,
        corpus_clusters,
        first_pages,
        rep_type,
        output_path=output_path,
        debug=debug,
    )
    # display_confusion_matrix(corpus_collections, corpus_clusters, debug)
    # calculate_cluster_precision(corpus_collections, corpus_clusters)
    calculate_scores_with_unknown_gold(corpus_vectorized, corpus_clusters, output_path=output_path)


def display_confusion_matrix(corpus_collections: List[str], corpus_clusters: List[int], debug: bool) -> None:
    le = preprocessing.LabelEncoder()
    corpus_collections_encoded = le.fit_transform(corpus_collections)
    unoptimized_cm = confusion_matrix(corpus_collections_encoded, corpus_clusters)

    if debug:
        # Show the confusion matrix before re-assignment below.
        debug_cm = sns.heatmap(unoptimized_cm, annot=True, fmt="d")
        debug_cm.show()

    # We don't know a priori which cluster corresponds to which collection. These next few steps attempt to pick
    # the best assignment of cluster to collection s.t. overall accuracy is maximized.
    # Ref: https://smorbieu.gitlab.io/accuracy-from-classification-to-clustering-evaluation/
    row_ind, col_ind = linear_sum_assignment(unoptimized_cm, maximize=True)
    # Linear sum assignment sorts by row by default, allowing you to re-arrange columns.
    # However, we want to sort by column so we can re-arrange the rows.
    assignments_sorted_by_col_ind = sorted(zip(row_ind, col_ind), key=lambda x: x[1])
    row_arrangement = [x[0] for x in assignments_sorted_by_col_ind]
    # Re-arrange the rows (i.e. assignments of collection to cluster) by the optimal arrangement.
    optimized_cm = unoptimized_cm[row_arrangement, :]

    fig = sns.heatmap(optimized_cm, annot=True, fmt="d")
    fig.set_xlabel("Cluster (Predicted)")
    fig.set_ylabel("Collection (Labeled)")
    plt.show()

    cluster_to_collection_index = dict(zip(col_ind, row_ind))
    collection_to_cluster_index = dict(zip(row_ind, col_ind))

    # Map the clusters into the same label indexing as the collections.
    corpus_clusters_remapped = np.array([cluster_to_collection_index[i] for i in corpus_clusters])
    report = classification_report(
        corpus_collections_encoded,
        corpus_clusters_remapped,
        target_names=[
            f"{le.classes_[k]} (Cluster {v})" if k < len(le.classes_) else f"n/a (Cluster {v})"
            for k, v in collection_to_cluster_index.items()
        ],
        zero_division=0,
    )
    logger.info(f"Classification Report\n{report}")


def calculate_cluster_precision(
    corpus_collections: List[str],
    corpus_clusters: List[int],
) -> None:
    """
    "Cluster precision" is an approximate measure of how well documents with the same layout are grouped together.
    It measures for a given cluster what fraction of the documents within it originated from the majority collection.
    The "majority collection" is the most frequent origin collection of all the documents in the cluster.

    For a given cluster, we typically expect that all the documents within it originated from the same collection.
    (This may not be the case if the user spreads documents with the same layout across different collections, but we
    assume that is not usually true.)

    Note that the macro average of cluster precision will trivially be 1.0 if the number of clusters == number of
    documents.
    """
    cluster_to_collections = defaultdict(list)
    for cluster, collection in zip(corpus_clusters, corpus_collections):
        cluster_to_collections[cluster].append(collection)

    cluster_precisions = {}
    for cluster, collections in sorted(cluster_to_collections.items()):
        _, majority_count = Counter(collections).most_common(1)[0]
        cluster_precisions[cluster] = majority_count / len(collections)

    logger.info(f"Cluster precision per-cluster:\n{pprint.pformat(cluster_precisions)}")
    logger.info(f"Macro average cluster precision: {statistics.mean(cluster_precisions.values())}")


def calculate_scores_with_unknown_gold(
    corpus_vectorized: List[int],
    corpus_clusters: List[int],
    output_path: Optional[Path] = None,
) -> None:
    silhouette = f"Silhouette coefficient: {silhouette_score(corpus_vectorized, corpus_clusters)}"
    ch = f"Calinski-Harabasz index: {calinski_harabasz_score(corpus_vectorized, corpus_clusters)}"

    logger.info(silhouette)
    logger.info(ch)

    if output_path:
        with open(output_path / "scores.txt", "w") as fh:
            fh.write(silhouette)
            fh.write("\n")
            fh.write(ch)


def get_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Cluster Impira documents")
    parser.add_argument(
        "-p",
        "--data-path",
        type=existing_directory,
        help="Path to directory containing collections. "
        "Each collection is assumed to be pre-processed using fetch_data.py",
        default="data/demo/",
    )
    parser.add_argument(
        "-m",
        "--models-path",
        type=existing_directory,
        help="Path to directory containing pretrained or finetuned models. ",
        default="finetuned_models/",
    )

    parser.add_argument(
        "-o",
        "--output-path",
        type=existing_directory,
        help="Path to save results in. ",
    )
    parser.add_argument(
        "-r",
        "--representation",
        type=str,
        help="Document representation",
        choices=[
            "rivlet_count",
            "rivlet_tfidf",
            "vanilla_lmv1",
            "finetuned_related_lmv1",
            "finetuned_unrelated_lmv1",
            "vanilla_lmv2",
        ],  # Must be a member of RepresentationType
        default="rivlet_count",
    )
    parser.add_argument(
        "-s",
        "--squash-strategy",
        type=str,
        help="Strategy to use for squashing hidden states",
        choices=[
            "average_all_words",
            "average_all_words_mask_pads",
            "last_word",
        ],
    )
    parser.add_argument(
        "-c",
        "--clustering-algorithm",
        type=str,
        help="Algorithm used to perform clustering",
        choices=["bayesian_gaussian_mixture"],  # Must be a member of ClusteringAlgorithm
        default="bayesian_gaussian_mixture",
    )
    parser.add_argument(
        "-k",
        "--num-clusters",
        type=int,
        help="Number of clusters to use",
        default=ClusteringParameters.N_COMPONENTS,
    )
    parser.add_argument(
        "-e",
        "--embedding-size",
        type=int,
        help="Maximum document embedding size",
        default=300,
    )
    parser.add_argument(
        "-n",
        "--normalize-length",
        action="store_true",
        help="Divide true sequence length by padded sequence length",
    )
    parser.add_argument(
        "--exclude-length",
        action="store_true",
        help="Don't include sequence length in embedding",
    )
    parser.add_argument("-d", "--debug", action="store_true")
    return parser


def cli_main():
    parser = get_parser()
    args = parser.parse_args()
    main(args)


if __name__ == "__main__":
    cli_main()
