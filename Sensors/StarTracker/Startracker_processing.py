import numpy as np
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation as R

from Sensors.StarTracker.gaia_catalog import (
    load_triangle_region,
    verify_fourth_star
)

# =========================================================
# CONFIG
# =========================================================

TOP_K_STARS = 8
MAX_TRIANGLES = 45
TRIANGLE_K_NEIGHBORS = 2

MIN_TRIANGLE_ANGLE_DEG = 2.0
MAX_TRIANGLE_ANGLE_DEG = 80.0

INLIER_THRESHOLD_DEG = 0.05
FAST_VERIFY_STARS = 1000


# =========================================================
# CACHE
# =========================================================

_catalog_tree_cache = {}

def _vecs_hash(vecs):
    # stable hash for caching; use a small digest of the bytes
    try:
        h = hash(vecs.tobytes())
    except Exception:
        # fallback to shape-based key if tobytes fails
        h = (vecs.shape, float(np.mean(vecs)))
    return h

def get_catalog_tree(vecs):
    key = _vecs_hash(vecs)
    if key not in _catalog_tree_cache:
        print("[CAT TREE] Building new KDTree for catalog vectors, N =", len(vecs))
        _catalog_tree_cache[key] = cKDTree(vecs.astype(np.float32))
    else:
        print("[CAT TREE] Reusing cached KDTree for catalog vectors, N =", len(vecs))
    return _catalog_tree_cache[key]




# =========================================================
# NORMALIZATION
# =========================================================

def normalize(v):
    v = np.asarray(v, dtype=np.float32)
    n = np.linalg.norm(v, axis=1, keepdims=True)
    return v / (n + 1e-9)


# =========================================================
# ANGLE MATRIX
# =========================================================

def angle_matrix(vecs):
    dot = vecs @ vecs.T
    np.clip(dot, -1.0, 1.0, out=dot)
    return np.arccos(dot)


# =========================================================
# ROBUST TRIANGLE SIGNATURE
# =========================================================

def triangle_signature(a, b, c):
    """
    Strong invariant triangle descriptor.

    OUTPUT (5D):
        [e1, e2, e3, e1/e2, e2/e3]
    """

    def ang(x, y):
        return np.arccos(np.clip(np.dot(x, y), -1.0, 1.0))

    e1 = ang(a, b)
    e2 = ang(b, c)
    e3 = ang(c, a)

    edges = np.array([e1, e2, e3], dtype=np.float32)
    edges.sort()

    edges = np.clip(edges, 1e-6, None)

    s = edges.sum()
    norm = edges / (s + 1e-9)

    r1 = edges[0] / (edges[1] + 1e-9)
    r2 = edges[1] / (edges[2] + 1e-9)

    return np.array([
        norm[0],
        norm[1],
        norm[2],
        r1,
        r2
    ], dtype=np.float32)


# =========================================================
# INLIER SCORING
# =========================================================

def count_inliers(rot_ci, cam_vecs, cat_vecs):
    cam_inertial = rot_ci.apply(cam_vecs)

    cat_tree = cKDTree(cat_vecs)
    dists, nn = cat_tree.query(cam_inertial, k=1)

    ang_thresh = np.radians(0.1)
    inliers = dists < ang_thresh

    print("\n[INLIER DEBUG FIXED]")
    print("total matches :", np.sum(inliers))
    print("mean error deg:", np.degrees(np.mean(dists)))
    print("max error deg :", np.degrees(np.max(dists)))

    return int(np.sum(inliers))


# =========================================================
# TRIANGLE BUILDING
# =========================================================

def build_triangles(vecs, flux):
    vecs = normalize(vecs)
    flux = np.asarray(flux)

    print("\n[BUILD_TRIANGLES]")
    print("Input stars:", len(vecs))
    print("Flux stats: min =", float(np.min(flux)), "max =", float(np.max(flux)))

    if len(vecs) < 3:
        print("Not enough stars for triangles.")
        return [], vecs

    if len(flux) > TOP_K_STARS:
        idx = np.argpartition(flux, -TOP_K_STARS)[-TOP_K_STARS:]
    else:
        idx = np.arange(len(flux))

    vecs = vecs[idx]
    n = len(vecs)

    print("Using TOP_K_STARS =", TOP_K_STARS, "=> selected", n, "stars for triangles")

    angles = angle_matrix(vecs)

    min_a = np.radians(MIN_TRIANGLE_ANGLE_DEG)
    max_a = np.radians(MAX_TRIANGLE_ANGLE_DEG)

    triangles = []
    angle_samples = []

    for i in range(n):
        for j in range(i + 1, n):
            a = angles[i, j]
            if a < min_a or a > max_a:
                continue

            for k in range(j + 1, n):
                b = angles[j, k]
                c = angles[k, i]

                if not (min_a <= b <= max_a):
                    continue
                if not (min_a <= c <= max_a):
                    continue

                angle_samples.extend([a, b, c])

                sig = triangle_signature(vecs[i], vecs[j], vecs[k])
                triangles.append((sig, (i, j, k)))

    if angle_samples:
        angle_samples = np.array(angle_samples)
        print("Triangle edge angle stats (deg):",
              "min =", np.degrees(np.min(angle_samples)),
              "max =", np.degrees(np.max(angle_samples)),
              "mean =", np.degrees(np.mean(angle_samples)))
    else:
        print("No valid triangle edges within angle constraints.")

    print("Total triangles built:", len(triangles), " (before MAX_TRIANGLES truncation)")

    return triangles[:MAX_TRIANGLES], vecs


# =========================================================
# ATTITUDE SOLVE
# =========================================================
def fast_kabsch(A, B):
    H = A.T @ B
    U, _, Vt = np.linalg.svd(H, full_matrices=False)

    Rmat = Vt.T @ U.T

    if np.linalg.det(Rmat) < 0:
        Vt[-1, :] *= -1
        Rmat = Vt.T @ U.T

    # orthogonality check
    ortho_err = np.linalg.norm(Rmat.T @ Rmat - np.eye(3))
    detR = np.linalg.det(Rmat)

    rot = R.from_matrix(Rmat)

    A_rot = rot.apply(A)
    residuals = np.linalg.norm(A_rot - B, axis=1)

    print("\n[FAST_KABSCH DEBUG]")
    print("A shape:", A.shape, "B shape:", B.shape)
    print("det(R):", detR)
    print("R orthogonality error:", ortho_err)
    print("Residual stats: min =", float(np.min(residuals)),
          "max =", float(np.max(residuals)),
          "mean =", float(np.mean(residuals)))

    # If orthogonality error is large or residuals are huge, warn and continue,
    # but mark the rotation as potentially bad by attaching an attribute.
    rot._kabsch_quality = {
        "ortho_err": float(ortho_err),
        "det": float(detR),
        "residual_mean": float(np.mean(residuals)),
        "residual_max": float(np.max(residuals))
    }

    return rot



def solve_attitude(cam_vecs, cat_vecs):
    cam_vecs = normalize(cam_vecs)
    cat_vecs = normalize(cat_vecs)

    print("\n[SOLVE_ATTITUDE]")
    print("cam_vecs:", cam_vecs.shape, "cat_vecs:", cat_vecs.shape)

    return fast_kabsch(cam_vecs, cat_vecs)


# =========================================================
# REFINEMENT
# =========================================================

def refine_solution(rot_ci, cam_vecs, cat_vecs):
    cam_vecs = normalize(cam_vecs)
    cat_vecs = normalize(cat_vecs)

    cat_in_cam = rot_ci.inv().apply(cat_vecs)
    tree = cKDTree(cat_in_cam)

    dists, idx = tree.query(cam_vecs, k=1)
    mask = dists < 0.01

    print("\n[REFINE DEBUG]")
    print("Refine inliers:", np.sum(mask))
    print("Refine distance stats: min =", float(np.min(dists)),
          "max =", float(np.max(dists)),
          "mean =", float(np.mean(dists)))

    if np.sum(mask) < 6:
        print("Not enough refine inliers, returning original rotation.")
        return rot_ci

    refined = fast_kabsch(
        cam_vecs[mask],
        cat_vecs[idx[mask]]
    )

    return refined


# =========================================================
# MATCHING CORE
# =========================================================

from collections import defaultdict

def match_triangles(
    cam_vecs,
    cam_flux,
    region_center,
    radius_deg=10.0
):

    print("\n[MATCH_TRIANGLES]")
    print("Region center (RA,DEC):", region_center, "radius_deg:", radius_deg)
    print("Camera stars (input):", len(cam_vecs))

    cat_hashes, cat_ids, cat_vecs, hash_tree = load_triangle_region(
        region_center,
        radius_deg
    )

    print("[CAT REGION DEBUG]")
    print("Loaded catalog from triangle region:")
    print("cat_hashes:", cat_hashes.shape,
          "cat_ids:", cat_ids.shape,
          "cat_vecs:", cat_vecs.shape)

    if len(cat_vecs) == 0:
        raise RuntimeError("Empty catalog")

    # sanity: cat_ids indices must be within cat_vecs
    if cat_ids.size > 0:
        max_id = np.max(cat_ids)
        min_id = np.min(cat_ids)
        print("cat_ids index range:", int(min_id), "to", int(max_id),
              "vs cat_vecs length:", len(cat_vecs))
        if max_id >= len(cat_vecs) or min_id < 0:
            raise RuntimeError("cat_ids out of range for cat_vecs")

    cat_vecs = normalize(cat_vecs)

    cat_hashes = np.asarray(cat_hashes, dtype=np.float32)

    print("\n[CATALOG HASH DEBUG]")
    print("cat_hashes shape:", cat_hashes.shape)

    if len(cat_hashes.shape) != 2:
        raise RuntimeError(f"Catalog hashes not 2D: {cat_hashes.shape}")

    if cat_hashes.shape[1] != 5:
        raise RuntimeError(
            f"Catalog triangle signature must be 5D, got {cat_hashes.shape[1]}"
        )

    norms = np.linalg.norm(cat_hashes, axis=1, keepdims=True)
    cat_hashes = cat_hashes / (norms + 1e-9)

    hash_tree = cKDTree(cat_hashes.astype(np.float32))
    cat_tree = cKDTree(cat_vecs.astype(np.float32))

    cam_triangles, cam_sel = build_triangles(cam_vecs, cam_flux)

    print("\n================ TRIANGLE DEBUG ================")
    print("Camera triangles:", len(cam_triangles))
    print("Selected stars   :", len(cam_sel))
    print("Raw camera stars :", len(cam_vecs))

    print("\n===== PYRAMID DEBUG =====")
    print("Catalog stars:", len(cat_vecs))
    print("Catalog triangles:", len(cat_hashes))
    print("Camera stars:", len(cam_vecs))
    print("Camera triangles:", len(cam_triangles))

    if len(cam_triangles) == 0:
        raise RuntimeError("No camera triangles built")

    vote_table = defaultdict(int)
    rot_votes = []

    sig_array = np.array([s for s, _ in cam_triangles])

    print("\n[CAMERA TRIANGLE SIG STATS]")
    print("mean:", np.mean(sig_array, axis=0))
    print("std :", np.std(sig_array, axis=0))

    for sig, tri in cam_triangles:
        sig = sig / (np.linalg.norm(sig) + 1e-9)
        dists, idx = hash_tree.query(sig, k=10)
        idx = np.atleast_1d(idx)

        print("\n[TRIANGLE MATCH DEBUG]")
        print("Camera triangle:", tri)
        print("Triangle hash distances:", dists)
        print("Best 3 catalog triangle indices:", idx[:3])

        cam_ids = list(tri)
        cam_tri = cam_sel[cam_ids]

        for i in idx:
            cat_tri_ids = cat_ids[i]
            cat_tri = cat_vecs[cat_tri_ids]

            rot = fast_kabsch(cam_tri, cat_tri)

            Rmat = rot.as_matrix()
            I = np.eye(3)

            print("\nRotation sanity:")
            print("det(R):", np.linalg.det(Rmat))
            print("orthogonality error:", np.linalg.norm(Rmat.T @ Rmat - I))

            rot_votes.append(rot)

            cam_in_cat = rot.apply(cam_vecs)
            dists_all, nn = cat_tree.query(cam_in_cat, k=1)

            inliers = dists_all < np.radians(0.1)

            print("[TRIANGLE ROT INLIERS] count:", int(np.sum(inliers)),
                  "min_err_deg:", float(np.degrees(np.min(dists_all))),
                  "max_err_deg:", float(np.degrees(np.max(dists_all))))

            if np.sum(inliers) < 10:
                continue

            num_inliers = int(np.sum(inliers))
            vote_table[num_inliers] += 1

    if not rot_votes:
        raise RuntimeError("No pyramid solution")

    scores = []

    for r in rot_votes:
        cam_in_cat = r.apply(cam_vecs)
        dists, _ = cat_tree.query(cam_in_cat, k=1)
        score = np.sum(dists < np.radians(0.1))
        scores.append(score)

    best_idx = int(np.argmax(scores))
    best_rot = rot_votes[best_idx]

    print("\n[PYRAMID SCORE DEBUG]")
    print("All scores:", scores)
    print("Best score index:", best_idx, "value:", scores[best_idx])

    cam_in_cat = best_rot.apply(cam_vecs)
    tree = cKDTree(cat_vecs)

    dists, nn = tree.query(cam_in_cat, k=1)
    mask = dists < np.radians(0.1)

    print("[PYRAMID FINAL MASK] inliers:", int(np.sum(mask)),
          "min_err_deg:", float(np.degrees(np.min(dists))),
          "max_err_deg:", float(np.degrees(np.max(dists))))

    if np.sum(mask) < 6:
        raise RuntimeError("No stable solution")

    refined = fast_kabsch(cam_vecs[mask], cat_vecs[nn[mask]])

    score = int(np.sum(mask))

    print("\n===== PYRAMID RESULT =====")
    print("Inliers:", score)

    boresight_cam = np.array([0.0, 0.0, 1.0], dtype=np.float32)
    boresight_cat = refined.apply(boresight_cam)

    ra_rec = np.degrees(np.arctan2(boresight_cat[1], boresight_cat[0])) % 360.0
    dec_rec = np.degrees(np.arcsin(boresight_cat[2]))

    print("\n[PYRAMID ATTITUDE DEBUG]")
    print("Recovered boresight (cat frame):", boresight_cat)
    print(f"Recovered RA  (from match_triangles): {ra_rec:.6f}")
    print(f"Recovered DEC (from match_triangles): {dec_rec:.6f}")

    return refined, score, cat_vecs


# =========================================================
# MAIN PIPELINE
# =========================================================

def process_star_tracker_output(data, region_center, radius_deg=10.0):

    print("STAR TRACKER PROCESSING INPUT")
    print("RegionCenter:", region_center)
    print("RadiusDeg:", radius_deg)
    print("Data:", data)

    cam_vecs_full = np.asarray(data["camera_vectors"], dtype=np.float32)
    cam_flux_full = np.asarray(data["flux"], dtype=np.float32)

    print("\n[PROCESS_PIPELINE]")
    print("Region center (RA,DEC):", region_center)
    print("Initial camera stars:", len(cam_vecs_full))
    print("Initial flux stats: min =", float(np.min(cam_flux_full)),
          "max =", float(np.max(cam_flux_full)))

    cam_vecs = cam_vecs_full.copy()
    cam_flux = cam_flux_full.copy()

    top_k = 300

    if len(cam_flux) > top_k:
       idx = np.argsort(cam_flux)[-top_k:]
       cam_vecs = cam_vecs[idx]
       cam_flux = cam_flux[idx]

    print("\n===== CAMERA FILTERING =====")
    print("Reduced camera stars to:", len(cam_vecs))
    print("Flux range:", float(np.min(cam_flux)), "to", float(np.max(cam_flux)))

    rot, score, cat_all = match_triangles(
        cam_vecs,
        cam_flux,
        region_center,
        radius_deg
    )

    rot = refine_solution(rot, cam_vecs, cat_all)

    print("\n================ ATTITUDE VALIDATION ================")

    boresight = rot.apply([0, 0, 1])

    ra_true_deg, dec_true_deg = region_center
    ra_true = np.radians(ra_true_deg)
    dec_true = np.radians(dec_true_deg)

    true_dir = np.array([
        np.cos(dec_true) * np.cos(ra_true),
        np.cos(dec_true) * np.sin(ra_true),
        np.sin(dec_true)
    ])

    dot = np.clip(np.dot(true_dir, boresight), -1.0, 1.0)
    ang_err_deg = np.degrees(np.arccos(dot))

    print("Recovered boresight:", boresight)
    print("True direction     :", true_dir)
    print("Dot product        :", dot)
    print("Angular error (deg):", ang_err_deg)

    final_score = count_inliers(rot, cam_vecs, cat_all)

    print("\n================ INLIER ANALYSIS ================")

    cam_inertial = rot.apply(cam_vecs)

    cat_tree = cKDTree(cat_all.astype(np.float32))
    dists, nn = cat_tree.query(cam_inertial, k=1)

    print("Total camera stars:", len(cam_vecs))
    print("Median error (deg):", np.degrees(np.median(dists)))
    print("90th percentile   :", np.degrees(np.percentile(dists, 90)))
    print("Max error         :", np.degrees(np.max(dists)))

    print("Inliers @0.05°:", np.sum(dists < np.radians(0.05)))
    print("Inliers @0.1° :", np.sum(dists < np.radians(0.1)))
    print("Inliers @0.5° :", np.sum(dists < np.radians(0.5)))

    boresight = rot.apply([0, 0, 1])

    ra = np.degrees(np.arctan2(boresight[1], boresight[0])) % 360
    dec = np.degrees(np.arcsin(boresight[2]))

    print("\n===== ATTITUDE DEBUG =====")
    print("Boresight vector:", boresight)
    print("True direction:", true_dir)
    print("Boresight dot product:", np.clip(np.dot(true_dir, boresight), -1.0, 1.0))
    print("Boresight angular error (deg):", ang_err_deg)

    return {
        "rotation_camera_to_inertial": rot,
        "quaternion": rot.as_quat(),
        "triangle_score": int(score),
        "inlier_score": int(final_score),
        "matched_stars": len(cam_vecs),
        "boresight_vector": boresight,
        "ra_deg": ra,
        "dec_deg": dec,
    }
