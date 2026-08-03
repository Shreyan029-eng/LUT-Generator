import cv2
import numpy as np
from PIL import Image
from rembg import remove

class CinematicColorGrader:
    def __init__(self, ref_path, tgt_path):
        self.ref_path = ref_path
        self.tgt_path = tgt_path
        print("Initializing AI & Math Engines...")

    def generate_mask(self, image_path):
        """Uses U-2-Net to generate binary masks for math, and feathered masks for blending."""
        print(f"  -> Generating AI mask for: {image_path}")
        img_bgr = cv2.imread(image_path)
        if img_bgr is None:
            raise FileNotFoundError(f"Could not read {image_path}. Check your path!")
        
        # Convert to PIL for the AI
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        mask_pil = remove(Image.fromarray(img_rgb), only_mask=True)
        
        # Threshold to pure black and white (0 or 255)
        _, binary_mask = cv2.threshold(np.array(mask_pil), 50, 255, cv2.THRESH_BINARY)
        
        # THE STRICT MASK: Erode heavily so no background pixels contaminate the foreground math
        kernel = np.ones((5, 5), np.uint8)
        strict_mask = cv2.erode(binary_mask, kernel, iterations=4)
        
        # THE FEATHERED MASK: Apply a wide blur to the strict mask for seamless visual compositing
        feathered_mask = cv2.GaussianBlur(strict_mask, (21, 21), 0)
        
        return strict_mask, feathered_mask

    def transfer_color(self, target_lab, t_mean, t_std, r_mean, r_std, luma_opacity=1.0, color_opacity=1.0):
        """Applies the Reinhard formula with opacity blending to act like a professional grading node."""
        t_l, t_a, t_b = cv2.split(target_lab)
        
        # LOWERED CLAMP: 1.2 prevents extreme contrast ratios from blowing out edge pixels
        std_ratio_l = min(r_std[0][0] / (t_std[0][0] + 1e-5), 1.2)
        std_ratio_a = min(r_std[1][0] / (t_std[1][0] + 1e-5), 1.2)
        std_ratio_b = min(r_std[2][0] / (t_std[2][0] + 1e-5), 1.2)
        
        # Calculate the raw math
        out_l_warped = (std_ratio_l * (t_l - t_mean[0][0])) + r_mean[0][0]
        out_a_warped = (std_ratio_a * (t_a - t_mean[1][0])) + r_mean[1][0]
        out_b_warped = (std_ratio_b * (t_b - t_mean[2][0])) + r_mean[2][0]
        
        # OPACITY BLENDING: Soften the math by mixing it with the original target pixels
        out_l = (out_l_warped * luma_opacity) + (t_l * (1.0 - luma_opacity))
        out_a = (out_a_warped * color_opacity) + (t_a * (1.0 - color_opacity))
        out_b = (out_b_warped * color_opacity) + (t_b * (1.0 - color_opacity))
        
        # Clip to valid LAB boundaries
        out_l = np.clip(out_l, 0, 255)
        out_a = np.clip(out_a, 0, 255)
        out_b = np.clip(out_b, 0, 255)
        
        return cv2.merge([out_l, out_a, out_b]).astype(np.float32)

    def process(self, output_path="final_cinematic_grade.jpg"):
        print("\n--- PHASE 1: AI Masking ---")
        # Now we grab BOTH masks from our function
        ref_mask_strict, ref_mask_feathered = self.generate_mask(self.ref_path)
        tgt_mask_strict, tgt_mask_feathered = self.generate_mask(self.tgt_path)

        print("\n--- PHASE 2: Color Space Profiling ---")
        ref_bgr = cv2.imread(self.ref_path)
        tgt_bgr = cv2.imread(self.tgt_path)

        # Convert to LAB space for mathematical shifting
        ref_lab = cv2.cvtColor(ref_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
        tgt_lab = cv2.cvtColor(tgt_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)

        # Invert the strict masks so we can target the backgrounds
        ref_mask_inv_strict = cv2.bitwise_not(ref_mask_strict)
        tgt_mask_inv_strict = cv2.bitwise_not(tgt_mask_strict)

        # Calculate Stats USING THE STRICT MASK to prevent edge contamination
        ref_fg_mean, ref_fg_std = cv2.meanStdDev(ref_lab, mask=ref_mask_strict)
        tgt_fg_mean, tgt_fg_std = cv2.meanStdDev(tgt_lab, mask=tgt_mask_strict)
        ref_bg_mean, ref_bg_std = cv2.meanStdDev(ref_lab, mask=ref_mask_inv_strict)
        tgt_bg_mean, tgt_bg_std = cv2.meanStdDev(tgt_lab, mask=tgt_mask_inv_strict)

        print("--- PHASE 3: Mathematical Color Warp ---")
        
        # FOREGROUND: Protect the subject! 
        # Keep 100% of his original lighting (0.0 luma) and apply a gentle 20% color tint
        graded_fg_lab = self.transfer_color(
            tgt_lab, tgt_fg_mean, tgt_fg_std, ref_fg_mean, ref_fg_std, 
            luma_opacity=0.0, color_opacity=0.2
        )
        
        # BACKGROUND: Apply the mood!
        # Keep 85% of original lighting so the wheat stays bright, but apply 50% color tint for the cinematic mood
        graded_bg_lab = self.transfer_color(
            tgt_lab, tgt_bg_mean, tgt_bg_std, ref_bg_mean, ref_bg_std, 
            luma_opacity=0.15, color_opacity=0.5
        )

        print("--- PHASE 4: Compositing & Export ---")
        # USE THE FEATHERED MASK to blend the two mathematical layers together smoothly
        mask_3d = cv2.cvtColor(tgt_mask_feathered, cv2.COLOR_GRAY2BGR).astype(np.float32) / 255.0

        # The Magic Blend
        final_lab = (graded_fg_lab * mask_3d) + (graded_bg_lab * (1.0 - mask_3d))
        
        final_bgr = cv2.cvtColor(final_lab.astype(np.uint8), cv2.COLOR_LAB2BGR)
        cv2.imwrite(output_path, final_bgr)
        print(f"\nSUCCESS! Check your folder for '{output_path}'")

if __name__ == "__main__":
    # Update these strings with your exact working absolute path!
    ref_img = r"D:\College\Summer Project  (1st Year)\Colorspace Analyzer\reference.png"
    tgt_img = r"D:\College\Summer Project  (1st Year)\Colorspace Analyzer\target2.jpg"
    out_img = r"D:\College\Summer Project  (1st Year)\Colorspace Analyzer\final_cinematic_grade.jpg"
    
    grader = CinematicColorGrader(ref_img, tgt_img)
    grader.process(out_img)