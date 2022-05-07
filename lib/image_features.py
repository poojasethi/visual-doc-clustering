from img2vec_pytorch import Img2Vec
from PIL import Image
from pathlib import Path


def get_image_features(image_path: Path, model: str = "resnet", cuda: bool = False):
    """
    Get a vectorized representation of a singe document image.
    """
    # Use resnet-18 by default
    if model == "resnet":
        model = "resnet-18"

    # Initialize Img2Vec with GPU
    img2vec = Img2Vec(model=model, cuda=cuda)

    # Read in an image (rgb format)
    img = Image.open(image_path).convert("RGB")

    # Get a vector from img2vec, returned as a torch FloatTensor
    vec = img2vec.get_vec([img], tensor=True)
    vec = vec.squeeze()

    return vec
