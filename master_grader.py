import argparse

import cv2
import numpy as np
from PIL import Image
from rembg import remove


def generate_mask(image_path):
    """Generate a strict (binary) and a feathered mask using the U-2-Net AI."""
    img_bgr = cv2.imread(image_path)
    if img_bgr is None:
        raise FileNotFoundError(f"Could not read {image_path}. Check your path!")

    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    mask_pil = remove(Image.fromarray(img_rgb), only_mask=True)

    _, binary_mask = cv2.threshold(np.array(mask_pil), 50, 255, cv2.THRESH_BINARY)

    # STRICT MASK: Erode heavily so no background pixels contaminate the foreground math
    kernel = np.ones((5, 5), np.uint8)
    strict_mask = cv2.erode(binary_mask, kernel, iterations=4)

    # FEATHERED MASK: Blur the strict mask for seamless visual compositing
    feathered_mask = cv2.GaussianBlur(strict_mask, (21, 21), 0)

    return strict_mask, feathered_mask


def apply_reinhard(target_lab, t_mean, t_std, r_mean, r_std, luma_opacity=1.0, color_opacity=1.0):
    """Vectorised Reinhard colour transfer with opacity blending (LAB float32 input)."""
    t_l, t_a, t_b = cv2.split(target_lab)

    # LOWERED CLAMP: 1.2 prevents extreme contrast ratios from blowing out edge pixels
    std_ratio_l = min(r_std[0][0] / (t_std[0][0] + 1e-5), 1.2)
    std_ratio_a = min(r_std[1][0] / (t_std[1][0] + 1e-5), 1.2)
    std_ratio_b = min(r_std[2][0] / (t_std[2][0] + 1e-5), 1.2)

    # Raw Reinhard warp
    out_l = (std_ratio_l * (t_l - t_mean[0][0])) + r_mean[0][0]
    out_a = (std_ratio_a * (t_a - t_mean[1][0])) + r_mean[1][0]
    out_b = (std_ratio_b * (t_b - t_mean[2][0])) + r_mean[2][0]

    # OPACITY BLENDING: Soften the math by mixing it with the original pixels
    out_l = (out_l * luma_opacity) + (t_l * (1.0 - luma_opacity))
    out_a = (out_a * color_opacity) + (t_a * (1.0 - color_opacity))
    out_b = (out_b * color_opacity) + (t_b * (1.0 - color_opacity))

    out_l = np.clip(out_l, 0, 255)
    out_a = np.clip(out_a, 0, 255)
    out_b = np.clip(out_b, 0, 255)

    return cv2.merge([out_l, out_a, out_b]).astype(np.float32)


def _mask_stats(lab, mask, fallback_mean, fallback_std):
    """Return masked mean/std, falling back to global stats when the mask is empty."""
    if mask is None or cv2.countNonZero(mask) == 0:
        return fallback_mean, fallback_std
    return cv2.meanStdDev(lab, mask=mask)


def grade_images(ref_path, tgt_path, output_path,
                 fg_luma_opacity=0.0, fg_color_opacity=0.2,
                 bg_luma_opacity=0.15, bg_color_opacity=0.5,
                 progress_callback=None):
    """Run the full masked cinematic grading pipeline and write the output image."""
    def report(message):
        if progress_callback:
            progress_callback(message)

    report("Masking reference image...")
    ref_mask_strict, ref_mask_feathered = generate_mask(ref_path)

    report("Masking target image...")
    tgt_mask_strict, tgt_mask_feathered = generate_mask(tgt_path)

    report("Profiling colour spaces...")
    ref_bgr = cv2.imread(ref_path)
    tgt_bgr = cv2.imread(tgt_path)

    ref_lab = cv2.cvtColor(ref_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
    tgt_lab = cv2.cvtColor(tgt_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)

    ref_global_mean, ref_global_std = cv2.meanStdDev(ref_lab)
    tgt_global_mean, tgt_global_std = cv2.meanStdDev(tgt_lab)

    # Invert the strict masks so we can target the backgrounds too
    ref_mask_inv = cv2.bitwise_not(ref_mask_strict)
    tgt_mask_inv = cv2.bitwise_not(tgt_mask_strict)

    # Stats WITH THE STRICT MASK to prevent edge contamination
    ref_fg_mean, ref_fg_std = _mask_stats(ref_lab, ref_mask_strict, ref_global_mean, ref_global_std)
    tgt_fg_mean, tgt_fg_std = _mask_stats(tgt_lab, tgt_mask_strict, tgt_global_mean, tgt_global_std)
    ref_bg_mean, ref_bg_std = _mask_stats(ref_lab, ref_mask_inv, ref_global_mean, ref_global_std)
    tgt_bg_mean, tgt_bg_std = _mask_stats(tgt_lab, tgt_mask_inv, tgt_global_mean, tgt_global_std)

    report("Applying colour warp...")
    # FOREGROUND: Protect the subject! Gentle color tint only.
    graded_fg_lab = apply_reinhard(tgt_lab, tgt_fg_mean, tgt_fg_std, ref_fg_mean, ref_fg_std,
                                   luma_opacity=fg_luma_opacity, color_opacity=fg_color_opacity)

    # BACKGROUND: Apply the mood!
    graded_bg_lab = apply_reinhard(tgt_lab, tgt_bg_mean, tgt_bg_std, ref_bg_mean, ref_bg_std,
                                   luma_opacity=bg_luma_opacity, color_opacity=bg_color_opacity)

    report("Compositing final image...")
    # USE THE FEATHERED MASK to blend the two mathematical layers together smoothly
    mask_3d = cv2.cvtColor(tgt_mask_feathered, cv2.COLOR_GRAY2BGR).astype(np.float32) / 255.0
    final_lab = (graded_fg_lab * mask_3d) + (graded_bg_lab * (1.0 - mask_3d))

    final_bgr = cv2.cvtColor(final_lab.astype(np.uint8), cv2.COLOR_LAB2BGR)
    cv2.imwrite(output_path, final_bgr)

    return output_path


def generate_lut(ref_path, tgt_path, output_path, lut_size=33,
                 luma_opacity=1.0, color_opacity=1.0, progress_callback=None):
    """Bake the Reinhard transfer into a standard 3D .cube LUT (blue channel fastest)."""
    if progress_callback:
        progress_callback("Baking 3D LUT (.cube)...")

    ref_bgr = cv2.imread(ref_path)
    tgt_bgr = cv2.imread(tgt_path)
    if ref_bgr is None or tgt_bgr is None:
        raise FileNotFoundError("Could not read an input image for LUT generation.")

    ref_lab = cv2.cvtColor(ref_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
    tgt_lab = cv2.cvtColor(tgt_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
    ref_mean, ref_std = cv2.meanStdDev(ref_lab)
    tgt_mean, tgt_std = cv2.meanStdDev(tgt_lab)

    # Build an RGB grid ordered red-slowest ... blue-fastest (matches the .cube spec)
    levels = np.linspace(0, 255, lut_size).astype(np.float32)
    r, g, b = np.meshgrid(levels, levels, levels, indexing="ij")
    rgb_grid = np.stack([r, g, b], axis=-1).reshape(-1, 1, 3).astype(np.uint8)

    lab_grid = cv2.cvtColor(rgb_grid, cv2.COLOR_RGB2LAB).astype(np.float32)
    graded_lab = apply_reinhard(lab_grid, tgt_mean, tgt_std, ref_mean, ref_std,
                                luma_opacity=luma_opacity, color_opacity=color_opacity)
    graded_rgb = cv2.cvtColor(graded_lab.astype(np.uint8), cv2.COLOR_LAB2RGB).astype(np.float32) / 255.0

    with open(output_path, "w", encoding="utf-8") as f:
        f.write('TITLE "Colorspace Analyzer LUT"\n')
        f.write(f"LUT_3D_SIZE {lut_size}\n")
        f.write("DOMAIN_MIN 0.0 0.0 0.0\n")
        f.write("DOMAIN_MAX 1.0 1.0 1.0\n\n")
        for i in range(graded_rgb.shape[0]):
            rr, gg, bb = graded_rgb[i, 0]
            f.write(f"{rr:.6f} {gg:.6f} {bb:.6f}\n")

    return output_path


def main():
    parser = argparse.ArgumentParser(description="Cinematic color grader with LUT export.")
    parser.add_argument("reference", help="Path to the reference image")
    parser.add_argument("target", help="Path to the target image")
    parser.add_argument("-o", "--output", default="final_cinematic_grade.jpg", help="Output image path")
    parser.add_argument("--lut", default=None, help="Optional .cube LUT output path")
    parser.add_argument("--lut-size", type=int, default=33, help="LUT size (default 33)")
    args = parser.parse_args()

    grade_images(args.reference, args.target, args.output)
    print(f"Graded image written to {args.output}")

    if args.lut:
        generate_lut(args.reference, args.target, args.lut, lut_size=args.lut_size)
        print(f"LUT written to {args.lut}")


if __name__ == "__main__":
    main()
