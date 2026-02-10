from typing import Any, Dict
from PIL import Image
from random import choice

# Pre-set options for the model to predict from
# in a real model these would be the output classes
FRUITS = ["Apple", "Banana", "Strawberry"]
FRESHNESS = ["fresh", "rotten"]

# will return a dict, something like:
# {"fruit": "Apple", "freshness": "fresh"}
def predict(img: Image.Image) -> Dict[str, Any]:
    # random for now
    return {
        "fruit": choice(FRUITS),
        "freshness": choice(FRESHNESS)
    }