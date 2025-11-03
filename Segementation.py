# iris_segmentation_gac_with_eyelids_improved.py
# Full corrected code - improved lower-eyelid detection, eyelash suppression,
# and smoothing of the limbus boundary (keeps the upper detection behavior intact).

import os
import cv2
import numpy as np
from skimage import img_as_float, measure, morphology
from skimage.filters import gaussian, sobel, scharr, threshold_otsu, unsharp_mask
from skimage.segmentation import morphological_geodesic_active_contour, clear_border
from scipy import ndimage
from scipy.signal import savgol_filter

# ---------- Utility ----------
def to_float_gray(bgr):
    g = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    return img_as_float(g)

def enhance_gray(gray):
    g8 = (gray*255).astype(np.uint8)
    g8 = cv2.bilateralFilter(g8, 7, 60, 60)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    g8 = clahe.apply(g8)
    g = g8.astype(np.float32)/255.0
    g = unsharp_mask(g, radius=1.0, amount=0.6)
    g = (g - np.percentile(g,1)) / (np.percentile(g,99)-np.percentile(g,1)+1e-6)
    return np.clip(g,0,1)

def circ_mask(shape, cx, cy, r):
    Y, X = np.ogrid[:shape[0], :shape[1]]
    return ((X-cx)**2 + (Y-cy)**2) <= r*r

def ring_roi(gray, cx, cy, r_in, r_out, pad=6):
    h,w = gray.shape
    x1 = max(0, int(cx - r_out - pad)); y1 = max(0, int(cy - r_out - pad))
    x2 = min(w, int(cx + r_out + pad)); y2 = min(h, int(cy + r_out + pad))
    return (slice(y1,y2), slice(x1,x2)), (x1,y1)

def eyelid_angular_mask(shape, cx, cy, top_deg=120, bot_deg=120, top_scale=0.35, bot_scale=0.5):
    h,w = shape
    Y,X = np.ogrid[:h,:w]
    ang = np.arctan2(Y-cy, X-cx)  # [-pi, pi]
    top = (ang > np.pi/6) & (ang < 5*np.pi/6)
    bot = (ang < -np.pi/6) & (ang > -5*np.pi/6)
    W = np.ones((h,w), np.float32)
    W[top] *= top_scale
    W[bot] *= bot_scale
    return W

def build_gimage(gray, cx, cy, pupil_r, eyelid_w=0.3, sigma=1.5, alpha=25.0, use_scharr=True):
    if use_scharr:
        grad = np.abs(scharr(gray))
    else:
        grad = np.abs(sobel(gaussian(gray, sigma=max(0.6, sigma*0.5))))
    gnorm = grad / (np.percentile(grad, 99.5) + 1e-6)
    gimage = 1.0 / (1.0 + (gnorm*alpha))
    lo, hi = np.percentile(gray, (5, 95))
    valid = (gray > lo) & (gray < hi)
    valid = morphology.remove_small_holes(valid, 1500)
    valid = morphology.remove_small_objects(valid, 1500)
    W = eyelid_angular_mask(gray.shape, cx, cy, top_scale=1.0-eyelid_w, bot_scale=1.0-eyelid_w*0.6)
    gimage = gimage * valid.astype(np.float32) * W.astype(np.float32)
    return np.clip(gimage, 0, 1)

def mgac_run(gimg, init_mask, iters, smoothing, threshold, balloon):
    ls = morphological_geodesic_active_contour(
        gimg, num_iter=iters, init_level_set=init_mask,
        smoothing=smoothing, threshold=threshold, balloon=balloon
    )
    return (ls > 0)

def boundary_strength(mask, grad):
    if not np.any(mask): return -1e9
    b = morphology.dilation(mask) ^ morphology.erosion(mask)
    if not np.any(b): return -1e9
    return float(np.mean(grad[b]))

def pick_largest_cc(mask):
    lbl = measure.label(mask)
    if lbl.max()==0: return mask
    areas = [(lab, np.sum(lbl==lab)) for lab in range(1, lbl.max()+1)]
    best = max(areas, key=lambda t:t[1])[0]
    return (lbl==best)

# ---------- Pupil ----------
def detect_pupil(gray):
    g = (gray*255).astype(np.uint8)
    clahe = cv2.createCLAHE(3.0,(8,8)).apply(g)
    thr = threshold_otsu(clahe/255.0)
    bin1 = (clahe/255.0) < thr*0.8
    bin1 = morphology.remove_small_objects(bin1, 200)
    bin1 = ndimage.binary_fill_holes(bin1)
    lbl = measure.label(bin1)
    if lbl.max()==0:
        h,w = gray.shape; return (w//2, h//2, min(h,w)//16)
    props = sorted(measure.regionprops(lbl), key=lambda p:p.area, reverse=True)
    for p in props[:5]:
        if p.perimeter>0:
            circ = 4*np.pi*p.area/(p.perimeter**2)
            if circ>0.55 and p.eccentricity<0.75:
                cy, cx = p.centroid
                r = np.sqrt(p.area/np.pi)
                return (float(cx), float(cy), float(r))
    p0 = props[0]
    cy, cx = p0.centroid; r = np.sqrt(p0.area/np.pi)
    return (float(cx), float(cy), float(r))

def segment_pupil_gac(gray, cx, cy, r):
    gimg = build_gimage(gray, cx, cy, r, eyelid_w=0.2, sigma=1.0, alpha=20.0)
    init = circ_mask(gray.shape, cx, cy, r*0.85)
    ls1 = mgac_run(gimg, init, iters=80, smoothing=1, threshold='auto', balloon=-2.0)
    ls2 = mgac_run(gimg, ls1, iters=40, smoothing=1, threshold='auto', balloon=-0.5)
    out = pick_largest_cc(ls2)
    return out

# ---------- Improved Eyelid Detection (suppresses eyelashes) ----------
def detect_eyelids(gray, cx, cy, pupil_r, limbus_r):
    """
    Returns a boolean mask that is True for pixels to KEEP (not clipped by eyelids).
    Strategy:
      - Upper ROI: morphological closing + contour filtering to ignore short eyelash edges.
      - Lower ROI: more lenient Canny, fallback to original method if needed.
      - Fit smooth parabolas to the largest contour(s).
    """
    h, w = gray.shape
    keep_mask = np.ones((h, w), dtype=bool)

    # --- Upper eyelid (preserve original good behavior, but suppress eyelash noise) ---
    y1 = max(0, int(cy - 1.25*limbus_r))
    y2 = int(cy)
    upper_roi = gray[y1:y2, :].copy()

    if upper_roi.size > 0:
        # preprocess to remove thin dark lashes
        roi_u = (upper_roi*255).astype(np.uint8)
        roi_u = cv2.medianBlur(roi_u, 5)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7,7))
        closed = cv2.morphologyEx(roi_u, cv2.MORPH_CLOSE, kernel, iterations=1)
        closed = cv2.GaussianBlur(closed, (5,5), 0)

        # edge detection (slightly stricter)
        edges_upper = cv2.Canny(closed, 40, 120)

        # find external contours and ignore small arcs (eyelashes)
        cnts, _ = cv2.findContours(edges_upper, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        long_cnts = [c for c in cnts if cv2.arcLength(c, False) > max(60, 0.08*w)]
        upper_coeff = None
        if long_cnts:
            # choose the largest arc (most likely eyelid curve)
            c = max(long_cnts, key=lambda cc: cv2.arcLength(cc, False))
            pts = c[:,0,:]  # (x, y) coordinates within ROI (x across columns, y across rows)
            xs = pts[:,0].astype(np.float32)
            ys = pts[:,1].astype(np.float32) + y1  # convert to global y
            # ensure adequate points and spread
            if len(xs) >= 25:
                try:
                    upper_coeff = np.polyfit(xs, ys, 2)
                except Exception:
                    upper_coeff = None
        else:
            upper_coeff = None
    else:
        upper_coeff = None

    # --- Lower eyelid (more tolerant) ---
    ly1 = int(cy)
    ly2 = min(h, int(cy + 1.5*limbus_r))
    lower_roi = gray[ly1:ly2, :].copy()
    lower_coeff = None
    if lower_roi.size > 0:
        roi_l = (lower_roi*255).astype(np.uint8)
        roi_l = cv2.medianBlur(roi_l, 5)
        # smaller kernel for closing (we don't want to oversmooth the lower boundary)
        kernel2 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5))
        closed_l = cv2.morphologyEx(roi_l, cv2.MORPH_CLOSE, kernel2, iterations=1)
        closed_l = cv2.GaussianBlur(closed_l, (3,3), 0)

        edges_lower = cv2.Canny(closed_l, 30, 90)
        pts_lower = np.column_stack(np.nonzero(edges_lower))
        if len(pts_lower) > 40:
            xs = pts_lower[:,1].astype(np.float32)
            ys = pts_lower[:,0].astype(np.float32) + ly1
            try:
                lower_coeff = np.polyfit(xs, ys, 2)
            except Exception:
                lower_coeff = None

    # --- Build keep_mask from fitted curves ---
    Y, X = np.mgrid[0:h, 0:w]
    if upper_coeff is not None:
        y_upper = np.polyval(upper_coeff, X)
        keep_mask[Y < y_upper] = False  # above the upper eyelid -> clip
    if lower_coeff is not None:
        y_lower = np.polyval(lower_coeff, X)
        keep_mask[Y > (y_lower + 4)] = False  # below lower eyelid -> clip (small downward offset)

    return keep_mask

# ---------- Smooth Limbus Contour ----------
def smooth_limbus_contour(mask, cx, cy, window=101, poly=3):
    """
    Smooth the boundary of a binary limbus mask by:
      - extracting the largest contour,
      - converting contour to polar coordinates around (cx,cy),
      - smoothing radius vs angle with Savitzky-Golay,
      - reconstructing a smooth polygon and filling it.
    """
    cnts = measure.find_contours(mask.astype(float), 0.5)
    if not cnts:
        return mask
    c = max(cnts, key=lambda a: a.shape[0])  # choose the longest contour
    xs = c[:,1].astype(np.float64)
    ys = c[:,0].astype(np.float64)
    ang = np.arctan2(ys - cy, xs - cx)
    ang = (ang + 2*np.pi) % (2*np.pi)
    order = np.argsort(ang)
    ang_s = ang[order]
    r = np.sqrt((xs - cx)**2 + (ys - cy)**2)[order]

    n = len(r)
    if n < 7:
        return mask

    # window must be odd and <= n
    w = min(window, n if n % 2 == 1 else n-1)
    if w < 5:
        w = 5 if n >= 5 else (n if n % 2 == 1 else n-1)

    pad = w // 2
    r_padded = np.concatenate([r[-pad:], r, r[:pad]])
    try:
        r_smooth_padded = savgol_filter(r_padded, w, poly)
    except Exception:
        # fallback: simple gaussian blur on radii
        r_smooth_padded = ndimage.gaussian_filter1d(r_padded, sigma=3)
    r_smooth = r_smooth_padded[pad:pad + n]

    # Rebuild smooth contour points
    x_smooth = cx + r_smooth * np.cos(ang_s)
    y_smooth = cy + r_smooth * np.sin(ang_s)
    pts = np.vstack((x_smooth, y_smooth)).T.astype(np.int32)

    # Ensure points are inside image bounds
    h, w_img = mask.shape
    pts[:,0] = np.clip(pts[:,0], 0, w_img-1)
    pts[:,1] = np.clip(pts[:,1], 0, h-1)

    poly = np.array([pts], dtype=np.int32)
    filled = np.zeros_like(mask, dtype=np.uint8)
    cv2.fillPoly(filled, poly, 1)
    filled = ndimage.binary_fill_holes(filled).astype(bool)
    return filled

# ---------- Limbus ----------
def segment_limbus_gac(gray, cx, cy, pupil_r):
    r_est = pupil_r*2.6
    sl, top_left = ring_roi(gray, cx, cy, pupil_r*1.1, r_est*1.35)
    g_crop = gray[sl].copy()
    # Note: coordinates inside g_crop are offset by top_left
    gx = cx - top_left[0]; gy = cy - top_left[1]

    gimg = build_gimage(g_crop, gx, gy, pupil_r, eyelid_w=0.35, sigma=1.2, alpha=28.0)
    # clear border to avoid spurious edges at crop seam
    gimg = clear_border(gimg < 1.0, buffer_size=2, bgval=0).astype(np.float32) * gimg

    seeds = [pupil_r*2.2, pupil_r*2.6, pupil_r*3.0]
    best_mask = None; best_score = -1e9
    grad_mag = np.abs(scharr(g_crop))

    for r0 in seeds:
        init = circ_mask(g_crop.shape, gx, gy, r0*0.85)
        s1 = mgac_run(gimg, init, iters=100, smoothing=2, threshold='auto', balloon=+0.6)
        s2 = mgac_run(gimg, s1,  iters=80,  smoothing=2, threshold='auto', balloon=+0.2)
        s3 = mgac_run(gimg, s2,  iters=60,  smoothing=2, threshold='auto', balloon= 0.0)
        cand = morphology.remove_small_holes(s3, 3000)
        cand = morphology.remove_small_objects(cand, 2500)
        cand = pick_largest_cc(cand)
        sc = boundary_strength(cand, grad_mag)
        if sc > best_score:
            best_score, best_mask = sc, cand

    full = np.zeros_like(gray, dtype=bool)
    if best_mask is None:
        return full
    full[sl] = best_mask
    full = morphology.binary_closing(full, morphology.disk(2))
    full = ndimage.binary_fill_holes(full)

    lim_props = measure.regionprops(full.astype(int))
    if lim_props:
        l = lim_props[0]; l_r = np.sqrt(l.area/np.pi)
        max_r = pupil_r*3.5
        if l_r > max_r:
            circ = circ_mask(gray.shape, cx, cy, max_r)
            full = full & circ

    full = full & (~circ_mask(gray.shape, cx, cy, pupil_r*1.1))
    return full

def segment_limbus_gac_with_eyelids(gray, cx, cy, pupil_r):
    limbus = segment_limbus_gac(gray, cx, cy, pupil_r)
    if not np.any(limbus):
        return limbus
    # Smooth limbus boundary (this will make upper limbus boundary smooth)
    limbus_sm = smooth_limbus_contour(limbus, cx, cy, window=101, poly=3)
    # Determine eyelid clipping mask
    lim_props = measure.regionprops(limbus_sm.astype(int))
    if not lim_props:
        eyelid_mask = np.ones_like(gray, dtype=bool)
    else:
        lr = np.sqrt(lim_props[0].area/np.pi)
        eyelid_mask = detect_eyelids(gray, cx, cy, pupil_r, lr)
    # Apply eyelid mask after smoothing (so eyelashes don't create jagged boundary)
    final = limbus_sm & eyelid_mask
    final = morphology.remove_small_objects(final, 500)
    final = ndimage.binary_fill_holes(final)
    return final

# ---------- Pipeline ----------
def segment_iris_gac_only(image_path):
    bgr = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if bgr is None:
        raise FileNotFoundError(image_path)
    gray = enhance_gray(to_float_gray(bgr))
    cx, cy, pr = detect_pupil(gray)
    pupil = segment_pupil_gac(gray, cx, cy, pr)
    props = measure.regionprops(pupil.astype(int))
    if props:
        cy, cx = props[0].centroid
        pr = np.sqrt(props[0].area/np.pi)
    limbus = segment_limbus_gac_with_eyelids(gray, cx, cy, pr)
    return bgr, gray, pupil, limbus

if __name__ == "__main__":
    # === EDIT THIS PATH ===
    IMAGE_PATH = r"C:\Users\CYBORG_15\Desktop\archive\MMU-Iris-Database\45\left\zaridahl5.bmp"
    # =======================
    bgr, gray, pupil, limbus = segment_iris_gac_only(IMAGE_PATH)

    vis = bgr.copy()
    # draw pupil contour (red) and limbus contour (green)
    for m, color in [(pupil,(0,0,255)), (limbus,(0,255,0))]:
        cnts = measure.find_contours(m.astype(float), 0.5)
        if cnts:
            c = max(cnts, key=lambda a: a.shape[0])
            pts = np.array([[int(xy[1]), int(xy[0])] for xy in c], np.int32)
            if pts.shape[0] >= 3:
                cv2.polylines(vis, [pts], True, color, 2, cv2.LINE_AA)

    cv2.imshow("GAC Iris with Eyelid Clipping (improved)", vis)
    cv2.waitKey(0)
    cv2.destroyAllWindows()