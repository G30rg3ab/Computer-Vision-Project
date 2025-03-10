from dataclasses import dataclass
import numpy as np_
import matplotlib.colors as mcolors


@dataclass
class DataSetConstants():
    class_intensitiy_dict = {'background':0, 'cat':38, 'dog':75}

@dataclass
class VisualisationConstants():
    # Define class colors

    class_colors = {
        0: "thistle",  # Background
        1: "red",     # Cat
        2: "green"    # Dog
    }
    # Create a color palette dynamically from class_colors
    palette = np_.zeros((256, 3), dtype=np_.uint8)
    for class_id, hex_color in class_colors.items():
        r, g, b = mcolors.hex2color(hex_color)  # Convert hex to (0-1) range
        palette[class_id] = [int(r * 255), int(g * 255), int(b * 255)]  # Convert to 0-255 scale

    palette[255] = [255, 255, 255]  # Ignore index - White

@dataclass
class BucketConstants():
    bucket = 'computer-vision-state-dictionaries'
    region = 'us-east-1'




