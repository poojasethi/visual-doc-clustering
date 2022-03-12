from logging import getLogger
from pathlib import Path
from typing import List, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.offsetbox import AnnotationBbox, OffsetImage
from matplotlib.transforms import Bbox
from PIL import Image
from sklearn.manifold import TSNE

logger = getLogger(__name__)


class ScatterKeys:
    VECTOR = "document_vector"
    PREDICTED = "predicted_collections"
    EXPECTED = "expected_collections"
    FIRST_PAGE = "first_page"
    CORRECT = "correct"
    SPLIT = "split"
    NEIGHBORS = "neighbors"
    DISTANCES = "distances"


def display_scatterplot_multilabel(
    data: pd.DataFrame,
    random_seed: Optional[int] = None,
) -> None:
    logger.info(f"Using the following random number seed: {random_seed}")

    doc_vectors = np.stack(data[ScatterKeys.VECTOR].apply(np.array))
    tsne = TSNE(verbose=1, random_state=random_seed)
    corpus_tsne = tsne.fit_transform(doc_vectors)

    x = corpus_tsne[:, 0]
    y = corpus_tsne[:, 1]
    alphas = data[ScatterKeys.SPLIT].apply(lambda s: 0.25 if s == "train" else 1).tolist()
    edgecolors = data[ScatterKeys.CORRECT].apply(lambda x: "red" if not x else "white")

    scatter = sns.scatterplot(
        x=x,
        y=y,
        hue=data[ScatterKeys.PREDICTED],
        style=data[ScatterKeys.EXPECTED],
        alpha=alphas,
        edgecolor=edgecolors,
        palette="Paired",
    )
    scatter.set_title("t-SNE visualization of supervised classifications")

    fig1 = scatter.get_figure()

    _make_scrollable_legend(scatter, fig1)

    first_pages = data[ScatterKeys.FIRST_PAGE].tolist()
    _add_first_page_animations(x, y, scatter, fig1, first_pages)

    _show_nearest_neighbors(x, y, data, scatter, fig1)

    plt.show()


def display_scatterplot(
    corpus_vectorized: List[List[int]],
    corpus_collections: List[str],
    corpus_clusters: List[int],
    first_pages: List[Path],
    rep_type: str,
    random_seed: Optional[int] = None,
    output_path: Optional[Path] = None,
) -> None:
    logger.info(f"Using the following random number seed: {random_seed}")
    tsne = TSNE(verbose=1, random_state=random_seed)
    corpus_tsne = tsne.fit_transform(corpus_vectorized)

    x = corpus_tsne[:, 0]
    y = corpus_tsne[:, 1]
    scatter = sns.scatterplot(x=x, y=y, hue=corpus_clusters, style=corpus_collections, palette="Paired")
    scatter.set_title(f"t-SNE visualization of documents encoded using: {rep_type}")
    fig1 = scatter.get_figure()

    _make_scrollable_legend(scatter, fig1)
    _add_first_page_animations(x, y, scatter, fig1, first_pages)
    if output_path:
        plt.savefig(output_path / "scatterplot.png")
    else:
        plt.show()


def _add_first_page_animations(x: List[int], y: List[int], scatter: Axes, fig: Figure, first_pages: List[Path]) -> None:
    # When we hover the mouse over a point, we want to be able to see a preview of the document.
    # The rest of the code in this function enables that. Code adapted from:
    # Ref: https://stackoverflow.com/questions/42867400/python-show-image-upon-hovering-over-a-point

    # Plot the points again so that lines can be used to find the points to place hover-over images.
    (lines,) = scatter.plot(x, y, ls="")

    # Create a placeholder annotation box. Its contents will be replaced with the actual images we'll want to show.
    arr = np.empty((len(x), 10, 10))
    im = OffsetImage(arr[0, :, :], zoom=1)
    xybox = (50.0, 50.0)
    ab = AnnotationBbox(
        im,
        (0, 0),
        xybox=xybox,
        xycoords="data",
        boxcoords="offset points",
        pad=0.3,
        arrowprops=dict(arrowstyle="->"),
    )
    scatter.add_artist(ab)
    ab.set_visible(False)

    # When a user hovers over a point, we use this callback to load and set the appropriate image.
    def _click(event):
        # If the mouse is over the scatter point
        if lines.contains(event)[0]:
            ind = lines.contains(event)[1]["ind"]
            ind = ind[0]

            # Get the figure size
            w, h = fig.get_size_inches() * fig.dpi
            ws = (event.x > w / 2.0) * -1 + (event.x <= w / 2.0)
            hs = (event.y > h / 2.0) * -1 + (event.y <= h / 2.0)

            # If event occurs in the top or right quadrant of the figure,
            # change the annotation box position relative to mouse.
            ab.xybox = (xybox[0] * ws, xybox[1] * hs)

            # Make annotation box visible
            ab.set_visible(True)

            # Place it at the position of the hovered scatter point
            ab.xy = (x[ind], y[ind])

            # Set the image corresponding to that point
            path = first_pages[ind]
            logger.info(f"Displaying first page of document from {path}")
            with Image.open(first_pages[ind]) as image:
                image.thumbnail((400, 400))
                im.set_data(image)
        else:
            # If the mouse is not over a scatter point
            ab.set_visible(False)
        fig.canvas.draw_idle()

    # Add callback for mouse moves.
    fig.canvas.mpl_connect("button_press_event", _click)


def _make_scrollable_legend(scatter: Axes, fig: Figure) -> None:
    scatter.legend(loc="upper left", bbox_to_anchor=(0.98, 0, 0.01, 1), fontsize="xx-small")
    d = {"down": 5, "up": -5}
    legend = scatter.get_legend()

    # When a user scrolls on the legend, we use this callback to allow it to scroll the legend accordingly.
    # Ref: https://stackoverflow.com/questions/55863590/adding-scroll-button-to-matlibplot-axes-legend
    def _scroll(event):
        if legend.contains(event):
            bbox = legend.get_bbox_to_anchor()
            bbox = Bbox.from_bounds(bbox.x0, bbox.y0 + d[event.button], bbox.width, bbox.height)
            tr = legend.axes.transAxes.inverted()
            legend.set_bbox_to_anchor(bbox.transformed(tr))
            fig.canvas.draw_idle()

    fig.canvas.mpl_connect("scroll_event", _scroll)


def _show_nearest_neighbors(x: List[int], y: List[int], data: pd.DataFrame, scatter: Axes, fig: Figure) -> None:
    """Show the neighbors that are used to predict which collection a document belongs to."""

    # Plot the points again so that lines can be used to find the points to place hover-over images.
    (lines,) = scatter.plot(x, y, ls="")

    arrows = []

    def _hover(event):
        if lines.contains(event)[0]:
            ind = lines.contains(event)[1]["ind"]
            ind = ind[0]

            x_coord, y_coord = x[ind], y[ind]

            row = data.iloc[ind, :]
            neighbors = row.loc[ScatterKeys.NEIGHBORS] or []

            for n in neighbors:
                x_coord_n, y_coord_n = x[n], y[n]
                arrow = scatter.annotate(
                    "",
                    xy=(x_coord, y_coord),
                    xytext=(x_coord_n, y_coord_n),
                    arrowprops=dict(fc="black", arrowstyle="simple"),
                )
                arrows.append(arrow)

        fig.canvas.draw_idle()

    def _on_key(event):
        # Press 'x' to clear the arrows.
        logger.info(event.key)
        if event.key == "x":
            for arrow in arrows:
                arrow.remove()
            arrows.clear()

    fig.canvas.mpl_connect("motion_notify_event", _hover)
    fig.canvas.mpl_connect("key_press_event", _on_key)
