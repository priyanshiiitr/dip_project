import cv2
import numpy as np
from skimage import restoration

# Blur Detection Using Laplacian Variance
def variance_of_laplacian(gray):
    return cv2.Laplacian(gray, cv2.CV_64F).var()

def detect_blur(gray, thresh=120.0):
    v = variance_of_laplacian(gray)
    return v, (v < thresh)

# Noise Estimation + Denoising
def denoise_image(img, sigma_thresh=10.0):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    mad = np.median(np.abs(gray - np.median(gray)))
    sigma = 1.4826 * mad

    if sigma > sigma_thresh:
        den = cv2.fastNlMeansDenoisingColored(img, None, 10, 10, 7, 21)
        return den, True

    return img, False

# Sharpening (Unsharp Mask)
def unsharp_mask(img, sigma=3, amount=1.5):
    blurred = cv2.GaussianBlur(img, (0,0), sigma)
    sharpened = cv2.addWeighted(img, 1.0 + amount, blurred, -amount, 0)
    return sharpened

# Contrast Enhancement using CLAHE
def apply_clahe_color(img):
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)

    clahe = cv2.createCLAHE(clipLimit=3.0)
    cl = clahe.apply(l)

    merged = cv2.merge((cl, a, b))
    final = cv2.cvtColor(merged, cv2.COLOR_LAB2BGR)
    return final

# Complete Enhancement Pipeline
def enhance_image(img):
    out = img.copy()
    gray = cv2.cvtColor(out, cv2.COLOR_BGR2GRAY)

    blur_val, is_blur = detect_blur(gray)

    out_steps = {
        "blur_value": float(blur_val),
        "is_blur": is_blur,
        "denoised": False,
        "sharpened": False,
        "clahe_applied": False
    }

    out, den = denoise_image(out)
    out_steps["denoised"] = den

    if is_blur:
        out = unsharp_mask(out)
        out_steps["sharpened"] = True

    out = apply_clahe_color(out)
    out_steps["clahe_applied"] = True

    return out, out_steps
