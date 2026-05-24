import numpy as np
from scipy.spatial.transform import Rotation as R
from StarTracker.gaia_catalog import load_gaia_region, load_triangle_region


# =========================================================
# Catalog builder
# =========================================================
def build_catalog(region_center, radius_deg=10.0):

    star_positions, fluxes = load_gaia_region(
        region_center=region_center,
        radius_deg=radius_deg
    )

    star_positions = star_positions / np.linalg.norm(
        star_positions, axis=1, keepdims=True
    )

    return star_positions, fluxes


# =========================================================
# Triangle invariant
# =========================================================
def triangle_signature(a, b, c):

    def ang(x, y):
        x = x / (np.linalg.norm(x) + 1e-12)
        y = y / (np.linalg.norm(y) + 1e-12)
        return np.arccos(np.clip(np.dot(x, y), -1.0, 1.0))

    sig = np.array([
        ang(a, b),
        ang(b, c),
        ang(c, a)
    ], dtype=np.float32)

    return np.sort(sig)   # ALWAYS (3,)


# =========================================================
# Build camera triangles
# =========================================================
def build_triangles(vecs, flux, top_k=25):

    idx = np.argsort(flux)[::-1][:top_k]
    vecs = vecs[idx]

    triangles = []
    n = len(vecs)

    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):

                sig = triangle_signature(vecs[i], vecs[j], vecs[k])
                triangles.append((sig, (i, j, k)))

    return triangles, vecs


# =========================================================
# Triangle matching
# =========================================================
def match_triangles(cam_vecs, cam_flux, region_center, radius_deg=10.0):

    # =====================================================
    # LOAD TRIANGLE DATABASE
    # =====================================================

    all_hashes, all_ids = load_triangle_region(region_center, radius_deg)

    best_err = np.inf
    best_cam = None
    best_cat_vecs = None

    # =====================================================
    # CAMERA TRIANGLES
    # =====================================================

    idx = np.argsort(cam_flux)[::-1][:25]
    cam_sel = cam_vecs[idx]

    cam_triangles = []
    n = len(cam_sel)

    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):

                sig = triangle_signature(cam_sel[i], cam_sel[j], cam_sel[k])
                cam_triangles.append((sig, (i, j, k)))

    # =====================================================
    # MATCH AGAINST CATALOG (DIRECT TILE USAGE)
    # =====================================================

    for tile_hashes, tile_ids in zip(all_hashes, all_ids):

        #data = load_triangle_region(tile_hashes)  # NOT RELOADED, just placeholder logic

        for cam_sig, cam_ids in cam_triangles:

            diff = np.linalg.norm(tile_hashes - cam_sig[None, :], axis=1)
            idx_best = np.argmin(diff)
            err = diff[idx_best]

            if err < best_err:

                best_err = err
                best_cam = cam_ids

                tri = tile_ids[idx_best]

                # =================================================
                # LOAD STAR VECTORS DIRECTLY FROM SAME TILE CACHE
                # =================================================
                star_vecs, _ = load_gaia_region(region_center, radius_deg)
                star_vecs = star_vecs / np.linalg.norm(star_vecs, axis=1, keepdims=True)

                best_cat_vecs = star_vecs[tri]

    # =====================================================
    # FINAL VALIDATION
    # =====================================================

    if best_cam is None or best_cat_vecs is None:
        raise RuntimeError("No triangle match found")

    best_cam = cam_sel[list(best_cam)]

    best_cat_vecs = np.asarray(best_cat_vecs)

    if best_cat_vecs.shape != (3, 3):
        raise RuntimeError(f"Bad cat_vecs shape: {best_cat_vecs.shape}")

    return best_cam, best_cat_vecs


# =========================================================
# ATTITUDE SOLVER
# =========================================================
def solve_attitude(cam_vecs, cat_vecs):

    assert cam_vecs.shape == (3, 3)
    assert cat_vecs.shape == (3, 3)

    rot, rms = R.align_vectors(cam_vecs, cat_vecs)

    return {
        "quaternion": rot.as_quat(),
        "rotation_matrix": rot.as_matrix(),
        "rms_error": rms
    }


# =========================================================
# PIPELINE
# =========================================================
def process_star_tracker_output(data, region_center, radius_deg=10.0):

    cam_vecs = np.asarray(data["camera_vectors"])
    cam_flux = np.asarray(data["flux"])

    cam_match, cat_match = match_triangles(
        cam_vecs,
        cam_flux,
        region_center,
        radius_deg
    )

    result = solve_attitude(cam_match, cat_match)

    result["matched_stars"] = 3

    return result