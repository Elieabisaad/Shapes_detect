# app.py
from flask import Flask, render_template, request, redirect, url_for
import os
import cv2
import numpy as np
from skimage import filters
from werkzeug.utils import secure_filename
import subprocess
import pandas as pd
import json
from heatmap_generator import generate_region_heatmap_overlay

app = Flask(__name__)
UPLOAD_FOLDER = 'static'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def prepare_img(img, prefix="", save_steps=False, output_dir="static",
                ksize1=9, sigma1=2.0, alpha=3.0,
                ksize2=21, sigma2=0, ksize3=5, sigma3=0,
                threshold=20):

    if img is None:
        raise ValueError(f"Image is empty in prepare_img() for {prefix}")

    step = lambda name: os.path.join(output_dir, 'intermediate', f"{prefix}_{name}.jpg")
    os.makedirs(os.path.join(output_dir, 'intermediate'), exist_ok=True)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    if save_steps: cv2.imwrite(step("1_gray"), gray)

    blurred = cv2.GaussianBlur(gray, (9, 9), 2.0)
    if save_steps: cv2.imwrite(step("2_blurred"), blurred)

    contrasted = cv2.convertScaleAbs(gray, alpha=3.0, beta=0)
    if save_steps: cv2.imwrite(step("3_contrasted"), contrasted)

    blur = cv2.GaussianBlur(contrasted, (21, 21), 0)
    highpass = cv2.subtract(contrasted, blur)
    if save_steps: cv2.imwrite(step("4_highpass"), highpass)

    result = cv2.normalize(highpass, None, 0, 255, cv2.NORM_MINMAX)
    result = cv2.GaussianBlur(result, (5, 5), 0)
    _, binary = cv2.threshold(result, 20, 255, cv2.THRESH_BINARY)
    if save_steps: cv2.imwrite(step("5_binary"), binary)

    return binary

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/refine', methods=['POST'])
def refine():
    full_image = request.form.get('full_image') == '1'
    save_intermediate = request.form.get('save_intermediate') == '1'

    def get_next_run_folder():
        base = "static"
        existing = [f for f in os.listdir(base) if f.startswith("run_") and os.path.isdir(os.path.join(base, f))]
        numbers = [int(f.split('_')[1]) for f in existing if f.split('_')[1].isdigit()]
        next_num = max(numbers) + 1 if numbers else 1
        next_folder = os.path.join(base, f"run_{next_num:03d}")
        os.makedirs(next_folder, exist_ok=True)
        return next_folder

    output_dir = get_next_run_folder()
    with open("static/latest_run.txt", "w") as f:
        f.write(output_dir)

    before_file = request.files['before_image']
    after_file = request.files['after_image']

    if not (before_file and allowed_file(before_file.filename)) or not (after_file and allowed_file(after_file.filename)):
        return "❌ Invalid file(s)", 400

    before_path = os.path.join(output_dir, 'uploaded_before.jpg')
    after_path = os.path.join(output_dir, 'uploaded_after.jpg')
    before_file.save(before_path)
    after_file.save(after_path)

    img_b = cv2.imread(before_path)
    img_a = cv2.imread(after_path)

    if img_a is None or img_b is None:
        return "❌ Failed to load uploaded images", 400

    if not full_image:
        try:
            x_start = int(request.form.get('x_start', 0))
            x_end = int(request.form.get('x_end', img_a.shape[1]))
            y_start = int(request.form.get('y_start', 0))
            y_end = int(request.form.get('y_end', img_a.shape[0]))
            img_a = img_a[y_start:y_end, x_start:x_end]
            img_b = img_b[y_start:y_end, x_start:x_end]
        except Exception as e:
            return f"❌ Crop failed: {e}", 400

    # ✅ Refinement parameters from form
    params = {
        "ksize1": int(request.form.get("ksize1", 9)),
        "sigma1": float(request.form.get("sigma1", 2.0)),
        "alpha": float(request.form.get("alpha", 3.0)),
        "ksize2": int(request.form.get("ksize2", 21)),
        "sigma2": float(request.form.get("sigma2", 0)),
        "ksize3": int(request.form.get("ksize3", 5)),
        "sigma3": float(request.form.get("sigma3", 0)),
        "threshold": int(request.form.get("threshold", 20))
    }

    a = prepare_img(img_a, prefix="after", save_steps=save_intermediate, output_dir=output_dir, **params)
    b = prepare_img(img_b, prefix="before", save_steps=save_intermediate, output_dir=output_dir, **params)

    fa = filters.frangi(a)
    fb = filters.frangi(b)

    fa_uint8 = (fa * 255).clip(0, 255).astype(np.uint8)
    fb_uint8 = (fb * 255).clip(0, 255).astype(np.uint8)

    cv2.imwrite(os.path.join(output_dir, 'Franjeafter.jpg'), fa_uint8)
    cv2.imwrite(os.path.join(output_dir, 'FranjeBefore.jpg'), fb_uint8)
    cv2.imwrite(os.path.join(output_dir, 'OriginalAfter.jpg'), img_a)
    cv2.imwrite(os.path.join(output_dir, 'OriginalBefor.jpg'), img_b)

    # Save metadata
    info = {
        "before_image": before_file.filename,
        "after_image": after_file.filename,
        "cropped": not full_image,
        **params
    }
    if not full_image:
        info["crop_box"] = {
            "x_start": x_start,
            "x_end": x_end,
            "y_start": y_start,
            "y_end": y_end
        }
    with open(os.path.join(output_dir, "info.json"), "w") as f:
        json.dump(info, f)

    return redirect(url_for('results'))


@app.route('/results')
def results():
    with open("static/latest_run.txt") as f:
        output_path = f.read().strip()
    output_dir = os.path.relpath(output_path, "static")
    analysis_done = os.path.exists(os.path.join(output_path, "analysis_status.txt"))
    summary_path = os.path.join(output_path, "summary.json")
    summary_data = {}
    if os.path.exists(summary_path):
        with open(summary_path) as f:
            summary_data = json.load(f)
    return render_template('results.html', analysis_done=analysis_done, output_dir=output_dir, summary=summary_data)
@app.route('/analyze', methods=['POST'])
def analyze_shapes():
    min_area = request.form.get('min_area', '10')
    with open("static/latest_run.txt") as f:
        output_dir = f.read().strip()
    result = subprocess.run(["python", "shape_analysis.py", "--min_area", str(min_area), "--output_dir", output_dir], capture_output=True, text=True)
    if result.returncode != 0:
        return f"<pre>❌ Error:\n{result.stderr}</pre>"
    return redirect(url_for('results'))

@app.route('/viewer', methods=['POST'])
def open_viewer():
    return redirect(url_for('viewer_page'))

@app.route('/viewer_page')
def viewer_page():
    cluster_id = request.args.get("cluster_id", default=1, type=int)
    with open("static/latest_run.txt") as f:
        output_path = f.read().strip()
    output_dir = os.path.relpath(output_path, "static")
    df = pd.read_excel(os.path.join(output_path, "cluster_shape_summary.xlsx"))
    max_cluster = len(df)
    return render_template('viewer.html', cluster_id=cluster_id, max_cluster=max_cluster, output_dir=output_dir)

@app.route('/results_table')
def results_table():
    with open("static/latest_run.txt") as f:
        output_path = f.read().strip()
    output_dir = os.path.relpath(output_path, "static")
    df = pd.read_excel(os.path.join(output_path, 'cluster_shape_summary.xlsx'))
    header = list(df.columns)
    rows = df.values.tolist()
    summary_path = os.path.join(output_path, "summary.json")
    summary_data = {}
    if os.path.exists(summary_path):
        with open(summary_path) as f:
            summary_data = json.load(f)
    return render_template('results_table.html', header=header, rows=rows, output_dir=output_dir, summary=summary_data)
@app.route('/heatmap', methods=['POST'])
def show_heatmap():
    try:
        with open("static/latest_run.txt") as f:
            output_path = f.read().strip()
        output_dir = os.path.relpath(output_path, "static")
        path = generate_region_heatmap_overlay(output_path)
        with open(os.path.join(output_path, "region_map.json")) as f:
            regions = json.load(f)
        filename = os.path.relpath(path, "static")
        return render_template("heatmap_view.html", image_url=url_for('static', filename=filename), regions=regions, output_dir=output_dir)
    except Exception as e:
        return f"<pre>❌ Heatmap generation failed:\n{e}</pre>"

@app.route("/cluster/<int:index>")
def view_cluster(index):
    return redirect(url_for('viewer_page', cluster_id=index))

if __name__ == '__main__':
    app.run(debug=True)
