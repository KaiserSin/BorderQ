import os
from pathlib import Path

import cv2
import gdown
import matplotlib.pyplot as plt
import numpy as np

# Fix LWCC weights path bug BEFORE importing LWCC
import lwcc.util.functions as lwcc_functions

def _patched_weights_check(model_name, model_weights):
    home = str(Path.home())
    weights_dir = os.path.join(home, ".lwcc", "weights")
    Path(weights_dir).mkdir(parents=True, exist_ok=True)
    
    filename = f"{model_name}_{model_weights}.pth"
    filepath = os.path.join(weights_dir, filename)
    
    if not os.path.isfile(filepath):
        url = lwcc_functions.build_url(filename)
        print(f"Downloading {filename}...")
        gdown.download(url, filepath, quiet=False)
    
    return filepath

lwcc_functions.weights_check = _patched_weights_check

# Import LWCC after patching
from lwcc import LWCC


def apply_density_boost(density: np.ndarray, boost: float = 1.5, threshold_percentile: float = 70):
    """Boost density in high-density areas."""
    threshold = np.percentile(density, threshold_percentile)
    boosted = density.copy()
    boosted[density > threshold] *= boost
    return boosted, boosted.sum()


def count_queue(image_path: str, model_name: str = "Bay", model_weights: str = "QNRF", 
                resize_img: bool = False, density_boost: float = 1.0):
    """Count people in queue using LWCC."""
    count, density = LWCC.get_count(
        image_path, 
        model_name=model_name, 
        model_weights=model_weights,
        return_density=True,
        resize_img=resize_img
    )
    
    if density_boost != 1.0:
        density, count = apply_density_boost(density, density_boost)
    
    return count, density


def visualize_density(image_path: str, density: np.ndarray, count: float, save_path: str = None):
    """Visualize density map overlay on image."""
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    axes[0].imshow(img)
    axes[0].set_title("Original")
    axes[0].axis("off")
    
    axes[1].imshow(density, cmap="jet")
    axes[1].set_title(f"Density Map\nCount: {count:.1f}")
    axes[1].axis("off")
    
    density_resized = cv2.resize(density, (img.shape[1], img.shape[0]))
    density_norm = (density_resized - density_resized.min()) / (density_resized.max() - density_resized.min() + 1e-8)
    
    axes[2].imshow(img)
    axes[2].imshow(density_norm, cmap="jet", alpha=0.5)
    axes[2].set_title(f"Overlay\nPeople: {count:.1f}")
    axes[2].axis("off")
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved: {save_path}")
    
    plt.show()


# Configuration
IMAGE_PATH = "resources/4 photo.png"
MODEL_NAME = "Bay"
MODEL_WEIGHTS = "QNRF"
DENSITY_BOOST = 1.5

# Run
print(f"Loading LWCC model ({MODEL_NAME} + {MODEL_WEIGHTS})...")
model = LWCC.load_model(model_name=MODEL_NAME, model_weights=MODEL_WEIGHTS)

print(f"Analyzing: {IMAGE_PATH}")
count, density = count_queue(IMAGE_PATH, resize_img=False, density_boost=DENSITY_BOOST)

print(f"\n{'='*50}")
print(f"LWCC Result ({MODEL_NAME}, {MODEL_WEIGHTS}, boost={DENSITY_BOOST})")
print(f"{'='*50}")
print(f"People count: {count:.1f}")
print(f"{'='*50}")

visualize_density(IMAGE_PATH, density, count, save_path="resources/lwcc_result.png")


