import argparse
import os
import random
from pathlib import Path
from typing import List, Tuple

import cv2
import numpy as np
import pandas as pd
from PIL import Image
import math


# ----------------------------
# CAD generation helpers
# ----------------------------
def generate_base_cad(size: int) -> Tuple[np.ndarray, List[Tuple[int, int]]]:
    """
    Create a binary CAD-like layout with orthogonal lines and via-like squares.
    Returns:
        cad (np.ndarray uint8): binary image {0,255}
        via_coords (list of (x,y)): centers of vias placed
    """
    cad = np.zeros((size, size), dtype=np.uint8)
    grid = size // 8
    rng_x = list(range(grid, size - grid, grid))
    rng_y = list(range(grid, size - grid, grid))
    random.shuffle(rng_x)
    random.shuffle(rng_y)
    num_lines = random.randint(3, 5)

    # Horizontal lines
    via_coords = []
    for y in rng_y[:num_lines]:
        w = random.randint(3, 6)
        cv2.rectangle(cad, (0, y - w // 2), (size - 1, y + w // 2), 255, -1)

    # Vertical lines
    for x in rng_x[:num_lines]:
        w = random.randint(3, 6)
        cv2.rectangle(cad, (x - w // 2, 0), (x + w // 2, size - 1), 255, -1)

    # Vias at intersections
    for x in rng_x[:num_lines]:
        for y in rng_y[:num_lines]:
            if random.random() < 0.35:
                v = random.randint(5, 9)
                cv2.rectangle(
                    cad,
                    (x - v // 2, y - v // 2),
                    (x + v // 2, y + v // 2),
                    255,
                    -1,
                )
                via_coords.append((x, y))
    return cad, via_coords


def inject_topological_defect(
    cad_img: np.ndarray, via_coords: List[Tuple[int, int]]
) -> Tuple[np.ndarray, str]:
    """
    Inject a topological defect: bridge, open, or missing_via.
    Returns modified CAD and defect_type.
    """
    cad = cad_img.copy()
    defect_type = random.choice(["bridge", "open", "missing_via"])
    h, w = cad.shape

    if defect_type == "bridge":
        y = random.randint(h // 4, 3 * h // 4)
        x0 = random.randint(w // 6, w // 2)
        x1 = x0 + random.randint(w // 8, w // 3)
        thickness = random.randint(4, 8)
        cv2.rectangle(cad, (x0, y - thickness // 2), (x1, y + thickness // 2), 255, -1)

    elif defect_type == "open":
        y = random.randint(h // 4, 3 * h // 4)
        x = random.randint(w // 4, 3 * w // 4)
        gap_w = random.randint(10, 18)
        gap_h = random.randint(6, 12)
        cv2.rectangle(cad, (x - gap_w // 2, y - gap_h // 2), (x + gap_w // 2, y + gap_h // 2), 0, -1)

    elif defect_type == "missing_via" and via_coords:
        x, y = random.choice(via_coords)
        size = random.randint(6, 10)
        cv2.rectangle(cad, (x - size // 2, y - size // 2), (x + size // 2, y + size // 2), 0, -1)
    else:
        defect_type = "bridge"  # fallback
        y = random.randint(h // 4, 3 * h // 4)
        x0 = random.randint(w // 6, w // 2)
        x1 = x0 + random.randint(w // 8, w // 3)
        thickness = random.randint(4, 8)
        cv2.rectangle(cad, (x0, y - thickness // 2), (x1, y + thickness // 2), 255, -1)

    return cad, defect_type


def apply_geometric_variation(cad_img: np.ndarray) -> np.ndarray:
    """
    Apply geometric/process variation without changing connectivity:
    mild dilation/erosion and rounding with blur+threshold.
    """
    cad = cad_img.copy()
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    if random.random() < 0.5:
        cad = cv2.dilate(cad, kernel, iterations=1)
    if random.random() < 0.5:
        cad = cv2.erode(cad, kernel, iterations=1)

    blur_sigma = random.uniform(0.6, 1.0)
    blurred = cv2.GaussianBlur(cad, (0, 0), blur_sigma)
    _, cad_thresh = cv2.threshold(blurred, 110, 255, cv2.THRESH_BINARY)
    cad = cad_thresh
    return cad


# ----------------------------
# Rendering stages
# ----------------------------
def physics_render(cad_img: np.ndarray) -> np.ndarray:
    """
    Simulate lithography/etch: density-dependent blur and bias.
    """
    cad_norm = cad_img.astype(np.float32) / 255.0
    blur_lo = cv2.GaussianBlur(cad_norm, (0, 0), 0.8)
    blur_hi = cv2.GaussianBlur(cad_norm, (0, 0), 1.6)

    density = cv2.GaussianBlur(cad_norm, (0, 0), 4.0)
    physics = blur_lo * (1 - density) + blur_hi * density

    bias = np.random.normal(0, 0.02)
    physics = np.clip(physics + bias, 0.0, 1.0)
    physics_uint8 = (physics * 255).astype(np.uint8)
    return physics_uint8


def simulate_sem_image(physics_img: np.ndarray) -> np.ndarray:
    """
    Add SEM-like appearance: noise, contrast jitter, fine roughness.
    """
    img = physics_img.astype(np.float32) / 255.0

    # Roughness texture
    rough = np.random.normal(0, 0.05, img.shape).astype(np.float32)
    rough = cv2.GaussianBlur(rough, (0, 0), 0.6)
    img = np.clip(img + 0.2 * rough, 0.0, 1.0)

    # Shot + Gaussian noise
    shot = np.random.poisson(img * 40) / 40.0
    gauss = np.random.normal(0, 0.02, img.shape).astype(np.float32)
    img = np.clip(0.7 * img + 0.3 * shot + gauss, 0.0, 1.0)

    # Contrast/brightness jitter
    contrast = random.uniform(0.85, 1.15)
    brightness = random.uniform(-0.05, 0.05)
    img = np.clip((img - 0.5) * contrast + 0.5 + brightness, 0.0, 1.0)

    # Vignette
    h, w = img.shape
    y, x = np.ogrid[:h, :w]
    dist = np.sqrt((x - w / 2) ** 2 + (y - h / 2) ** 2)
    dist_norm = dist / dist.max()
    vignette = 1 - 0.12 * dist_norm
    img = np.clip(img * vignette, 0.0, 1.0)

    return (img * 255).astype(np.uint8)


# ----------------------------
# Dataset generation
# ----------------------------
def decide_class() -> str:
    """
    Decide class with ~50% clean, 25% geom_var, 25% topo_defect.
    """
    r = random.random()
    if r < 0.5:
        return "clean"
    if r < 0.75:
        return "geom_var"
    return "topo_defect"


def assign_split(train_ratio: float, val_ratio: float, test_ratio: float) -> str:
    r = random.random()
    if r < train_ratio:
        return "train"
    if r < train_ratio + val_ratio:
        return "val"
    return "test"


def save_image(array: np.ndarray, path: Path) -> None:
    Image.fromarray(array).save(path)


def generate_dataset(
    out_dir: Path,
    num_samples: int,
    image_size: int,
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
) -> None:
    """
    Main loop: generate CAD -> physics -> SEM, inject defects/variations,
    save images and metadata.
    """
    for sub in ["cad", "physics", "sem"]:
        (out_dir / sub).mkdir(parents=True, exist_ok=True)

    records = []
    for i in range(num_samples):
        cls = decide_class()
        split = assign_split(train_ratio, val_ratio, test_ratio)
        cad, via_coords = generate_base_cad(image_size)
        defect_type = "none"

        if cls == "geom_var":
            cad = apply_geometric_variation(cad)
        elif cls == "topo_defect":
            cad, defect_type = inject_topological_defect(cad, via_coords)
            cad = apply_geometric_variation(cad)

        physics = physics_render(cad)
        sem = simulate_sem_image(physics)

        cad_path = out_dir / "cad" / f"cad_{i}.png"
        physics_path = out_dir / "physics" / f"physics_{i}.png"
        sem_path = out_dir / "sem" / f"sem_{i}.png"

        save_image(cad, cad_path)
        save_image(physics, physics_path)
        save_image(sem, sem_path)

        records.append(
            {
                "id": i,
                "split": split,
                "class": cls,
                "defect_type": defect_type,
                "cad_path": str(cad_path),
                "physics_path": str(physics_path),
                "sem_path": str(sem_path),
            }
        )

        if (i + 1) % 500 == 0:
            print(f"[+] Generated {i + 1}/{num_samples}")

    df = pd.DataFrame(records)
    df.to_csv(out_dir / "metadata.csv", index=False)
    print(f"Done. Metadata saved to {out_dir / 'metadata.csv'}")


# ----------------------------
# CLI
# ----------------------------
def parse_args():
    parser = argparse.ArgumentParser(description="Generate synthetic D2DB dataset.")
    parser.add_argument("--out_dir", type=str, required=True, help="Output directory")
    parser.add_argument("--num_samples", type=int, default=20000, help="Total samples")
    parser.add_argument("--image_size", type=int, default=128, help="Square image size")
    parser.add_argument("--train_ratio", type=float, default=0.7, help="Train split fraction")
    parser.add_argument("--val_ratio", type=float, default=0.15, help="Validation split fraction")
    parser.add_argument("--test_ratio", type=float, default=0.15, help="Test split fraction")
    parser.add_argument("--seed", type=int, default=123, help="Random seed")
    return parser.parse_args()


def main():
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)

    if not math.isclose(args.train_ratio + args.val_ratio + args.test_ratio, 1.0, rel_tol=1e-3):
        raise ValueError("train_ratio + val_ratio + test_ratio must sum to 1.0")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    generate_dataset(
        out_dir=out_dir,
        num_samples=args.num_samples,
        image_size=args.image_size,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
    )


if __name__ == "__main__":
    main()
