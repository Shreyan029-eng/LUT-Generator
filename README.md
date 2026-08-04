# LUT-Generator
Colorspace analyzer and LUT generator, used to extract color themes from images and apply it to another image on the spot, or generate a LUT for future applications in video and photographs.

## Features
- Upload a **reference image** (the look) and a **target image** (your shot) in a web UI.
- AI-driven subject/background separation (U-2-Net) so the subject and background are graded with different strengths.
- Optional **3D LUT (.cube)** export to reuse the grade in Resolve, Premiere, etc.
- Adjustable foreground / background color strength, configurable LUT resolution.
- Live progress bar + status messages while the AI pipeline runs.

## Quick Start

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Run the web app
```bash
python app.py
```
Open http://127.0.0.1:5000

- Drag or click to upload both images, toggle **Generate 3D LUT (.cube)**, tweak the strength sliders, then hit **Grade Images**.
- On success you can download the graded image and, if requested, the `.cube` LUT.

### 3. Run from the command line (no UI)
```bash
python master_grader.py reference.png target2.jpg -o graded.jpg --lut grade.cube --lut-size 33
```

## Project Structure
```
app.py               Flask backend (upload / process / progress / result API)
master_grader.py     Core grading engine (masking, Reinhard transfer, LUT baking, CLI)
templates/index.html Web UI
static/style.css     Styling
static/script.js     Frontend logic (upload, polling, results)
uploads/             Uploaded input images (created at runtime, gitignored)
outputs/             Graded images + .cube LUTs (created at runtime, gitignored)
```

## How It Works
1. **AI Masking** — U-2-Net (`rembg`) segments the subject out of both images. A *strict* mask (eroded) drives the math; a *feathered* mask drives the compositing.
2. **Color Space Profiling** — images are converted to CIE L\*a\*b\*, and the mean/std of the foreground and background are computed per image.
3. **Color Warp** — a Reinhard-style transfer shifts the target's statistics toward the reference's, with opacity blending. The subject gets a gentle tint (default 20%), the background gets the mood (default 50%), then both are blended with the feathered mask.
4. **LUT Baking** — the same transfer is evaluated across a regular RGB grid and written to a standard `.cube` file (blue channel fastest), ready for any editor.

## Troubleshooting
- **First run is slow** — the U-2-Net model (~170 MB) is downloaded once on first use; subsequent runs are much faster.
- **`cublasLt64_13.dll ... FAIL` warnings from onnxruntime** — harmless. onnxruntime falls back to CPU when CUDA isn't installed.
- **No subject detected** — if an image has no clear subject, the grader falls back to whole-image statistics instead of failing.
- Port 5000 busy? Change `app.run(..., port=5000)` at the bottom of `app.py`.
