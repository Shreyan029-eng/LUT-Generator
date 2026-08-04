import os
import threading
import uuid

from flask import Flask, jsonify, render_template, request, send_from_directory

import master_grader as grader

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_DIR = os.path.join(BASE_DIR, "uploads")
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

ALLOWED_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff"}

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 50 * 1024 * 1024  # 50 MB upload limit

JOBS = {}
JOBS_LOCK = threading.Lock()
PROCESS_LOCK = threading.Lock()
MAX_JOBS = 100


def _clamp(value, low, high, default):
    try:
        return max(low, min(high, float(value)))
    except (TypeError, ValueError):
        return default


def _phase_pct(message):
    mapping = {
        "Warming up AI model": 5,
        "Masking reference image": 20,
        "Masking target image": 40,
        "Profiling colour spaces": 55,
        "Applying colour warp": 65,
        "Compositing final image": 80,
        "Baking 3D LUT": 90,
        "Done": 100,
    }
    for key, pct in mapping.items():
        if message.startswith(key):
            return pct
    return None


def _set_progress(job_id, message, pct=None):
    with JOBS_LOCK:
        JOBS[job_id]["message"] = message
        if pct is not None:
            JOBS[job_id]["progress"] = pct


def _run_job(job_id, ref_path, tgt_path, options):
    # PROCESS_LOCK serialises jobs: rembg's first-run model download must not run twice.
    with PROCESS_LOCK:
        try:
            _set_progress(job_id, "Warming up AI model (first run may download weights)...", 5)

            out_img = os.path.join(OUTPUT_DIR, f"{job_id}_graded.jpg")
            grader.grade_images(
                ref_path, tgt_path, out_img,
                fg_color_opacity=options["fg_color_opacity"],
                bg_color_opacity=options["bg_color_opacity"],
                progress_callback=lambda msg: _set_progress(job_id, msg, _phase_pct(msg)),
            )

            result = {"image_url": f"/outputs/{os.path.basename(out_img)}"}

            if options["generate_lut"]:
                lut_path = os.path.join(OUTPUT_DIR, f"{job_id}.cube")
                grader.generate_lut(
                    ref_path, tgt_path, lut_path,
                    lut_size=options["lut_size"],
                    progress_callback=lambda msg: _set_progress(job_id, msg, _phase_pct(msg)),
                )
                result["lut_url"] = f"/outputs/{os.path.basename(lut_path)}"

            with JOBS_LOCK:
                JOBS[job_id]["status"] = "done"
                JOBS[job_id]["result"] = result
            _set_progress(job_id, "Done", 100)
        except Exception as exc:
            with JOBS_LOCK:
                JOBS[job_id]["status"] = "error"
                JOBS[job_id]["error"] = str(exc)
            _set_progress(job_id, f"Error: {exc}")


def _save_file(file_storage, job_id, role):
    ext = os.path.splitext(file_storage.filename)[1].lower()
    path = os.path.join(UPLOAD_DIR, f"{job_id}_{role}{ext}")
    file_storage.save(path)
    return path


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/outputs/<path:filename>")
def outputs(filename):
    return send_from_directory(OUTPUT_DIR, filename)


@app.route("/api/process", methods=["POST"])
def process():
    ref_file = request.files.get("reference")
    tgt_file = request.files.get("target")
    if ref_file is None or tgt_file is None:
        return jsonify({"error": "Both a reference and a target image are required."}), 400

    ref_ext = os.path.splitext(ref_file.filename)[1].lower()
    tgt_ext = os.path.splitext(tgt_file.filename)[1].lower()
    if ref_ext not in ALLOWED_EXTENSIONS or tgt_ext not in ALLOWED_EXTENSIONS:
        return jsonify({"error": "Unsupported file type. Allowed: PNG, JPG, JPEG, WEBP, BMP, TIFF."}), 400

    job_id = uuid.uuid4().hex
    ref_path = _save_file(ref_file, job_id, "ref")
    tgt_path = _save_file(tgt_file, job_id, "tgt")

    options = {
        "generate_lut": request.form.get("generate_lut") == "true",
        "fg_color_opacity": _clamp(request.form.get("fg_color_opacity"), 0.0, 1.0, 0.2),
        "bg_color_opacity": _clamp(request.form.get("bg_color_opacity"), 0.0, 1.0, 0.5),
        "lut_size": int(_clamp(request.form.get("lut_size"), 2, 129, 33)),
    }

    with JOBS_LOCK:
        JOBS[job_id] = {
            "status": "running",
            "message": "Queued",
            "progress": 0,
            "result": None,
            "error": None,
        }
        while len(JOBS) > MAX_JOBS:
            JOBS.pop(next(iter(JOBS)))

    threading.Thread(target=_run_job, args=(job_id, ref_path, tgt_path, options), daemon=True).start()
    return jsonify({"job_id": job_id})


@app.route("/api/progress/<job_id>")
def progress(job_id):
    with JOBS_LOCK:
        job = JOBS.get(job_id)
    if job is None:
        return jsonify({"status": "not_found"}), 404
    return jsonify(job)


@app.route("/api/result/<job_id>")
def result(job_id):
    with JOBS_LOCK:
        job = JOBS.get(job_id)
    if job is None:
        return jsonify({"status": "not_found"}), 404
    return jsonify(job)


if __name__ == "__main__":
    app.run(debug=True, threaded=True, host="127.0.0.1", port=5000)
