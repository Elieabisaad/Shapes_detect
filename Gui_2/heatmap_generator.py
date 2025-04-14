# heatmap_generator.py
import cv2
import numpy as np
import pandas as pd
from skimage import measure, morphology, filters
from skimage.morphology import footprint_rectangle
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
import matplotlib
import json
import os

matplotlib.use('Agg')

def generate_region_heatmap_overlay(output_dir="static"):
    frangi_path = os.path.join(output_dir, 'FranjeBefore.jpg')
    excel_path = os.path.join(output_dir, 'cluster_shape_summary.xlsx')
    output_combined_path = os.path.join(output_dir, 'region_heatmap_with_colorbar.jpg')
    json_output_path = os.path.join(output_dir, 'region_map.json')

    frangi = cv2.imread(frangi_path, cv2.IMREAD_GRAYSCALE)
    if frangi is None:
        raise FileNotFoundError(f"❌ '{frangi_path}' not found!")

    df = pd.read_excel(excel_path)
    df['Diff'] = df['After Shape Count'] - df['Before Shape Count']
    shape_diffs = df['Diff'].values

    thresh_val = filters.threshold_otsu(frangi)
    binary = frangi > thresh_val
    closed = morphology.closing(binary, footprint_rectangle((5, 5)))
    labeled = measure.label(closed)
    regions = measure.regionprops(labeled)

    positive_diffs = np.clip(shape_diffs, 0, None)
    norm_diffs = cv2.normalize(positive_diffs.astype(np.float32), None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    heatmap_mask = np.zeros((frangi.shape[0], frangi.shape[1], 3), dtype=np.uint8)
    region_metadata = []

    for idx, region in enumerate(regions):
        if idx >= len(norm_diffs):
            break
        coords = region.coords
        minr, minc, maxr, maxc = region.bbox
        cx, cy = minc + (maxc - minc) // 2, minr + (maxr - minr) // 2

        # Get shape change value
        diff = int(df.loc[idx, 'Diff'])
        before = int(df.loc[idx, 'Before Shape Count'])
        after = int(df.loc[idx, 'After Shape Count'])

        if diff < 0:
            color_bgr = np.array([255, 255, 255], dtype=np.uint8)  # white for loss
        else:
            color_bgr = cv2.applyColorMap(np.array([[norm_diffs[idx]]], dtype=np.uint8), cv2.COLORMAP_TURBO)[0, 0]

        color_rgb = color_bgr[::-1]  # ✅ Convert from BGR to RGB

        region_metadata.append({
            "id": idx,
            "center": [cx, cy],
            "before": before,
            "after": after,
            "diff": diff,
            "color": color_rgb.tolist()  # ✅ Correct RGB stored
        })


        for r, c in coords:
            heatmap_mask[r, c] = color_bgr

    with open(json_output_path, 'w') as f:
        json.dump(region_metadata, f)

    blurred = cv2.GaussianBlur(heatmap_mask, (15, 15), sigmaX=0)
    frangi_bgr = cv2.cvtColor(frangi, cv2.COLOR_GRAY2BGR)
    final = cv2.addWeighted(frangi_bgr, 0.7, blurred, 0.6, 0)
    cv2.imwrite(os.path.join(output_dir, "_temp_result.jpg"), final)

    fig, ax = plt.subplots(figsize=(1, 6))
    cmap = plt.get_cmap('turbo')
    norm = Normalize(vmin=np.min(positive_diffs), vmax=np.max(positive_diffs))
    cb = plt.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), cax=ax)
    cb.set_label("Shape Increase", fontsize=10)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "_colorbar.jpg"), dpi=150, bbox_inches='tight', pad_inches=0.1)
    plt.close()

    img = cv2.imread(os.path.join(output_dir, "_temp_result.jpg"))
    bar = cv2.imread(os.path.join(output_dir, "_colorbar.jpg"))
    bar_resized = cv2.resize(bar, (int(img.shape[0] * 0.1), img.shape[0]))
    combined = cv2.hconcat([img, bar_resized])
    cv2.imwrite(output_combined_path, combined)

    return output_combined_path
