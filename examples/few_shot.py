from PIL import Image
import json

few_shot_data = [
    {
        "image": "./examples/images/20_meters.jpg",
        "text": "This is how it looks like from 20 meters away",
    },
    {
        "image": "./examples/images/50_meters.jpg",
        "text": "This is how it looks like from 50 meters away",
    },
    {
        "image": "./examples/images/100_meters.jpg",
        "text": "This is how it looks like from 100 meters away",
    },
]
