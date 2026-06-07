import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R
import time

from Sensors.StarTracker.gaia_catalog import load_gaia_region
from Sensors.StarTracker.constellations import (
    BIG_DIPPER_STARS,
    BIG_DIPPER_LINES,
    get_constellation_vectors
)

# =========================================================
# GLOBAL PSF SETUP (kept outside for performance)
# =========================================================

def build_psf(image_size):
    RES_SCALE = image_size / 1024.0
    PSF_SIGMA = 1.1 * RES_SCALE
    PSF_RADIUS = int(13 * RES_SCALE)

    yy, xx = np.mgrid[
        -PSF_RADIUS:PSF_RADIUS + 1,
        -PSF_RADIUS:PSF_RADIUS + 1
    ]

    kernel = np.exp(-(xx**2 + yy**2) / (2.0 * PSF_SIGMA**2)).astype(np.float32)
    return kernel, PSF_RADIUS


def add_star_psf(img, x0, y0, flux, kernel, r):
    h, w = img.shape

    x1 = max(0, x0 - r)
    x2 = min(w, x0 + r + 1)
    y1 = max(0, y0 - r)
    y2 = min(h, y0 + r + 1)

    kx1 = x1 - (x0 - r)
    kx2 = kx1 + (x2 - x1)

    ky1 = y1 - (y0 - r)
    ky2 = ky1 + (y2 - y1)

    img[y1:y2, x1:x2] += flux * kernel[ky1:ky2, kx1:kx2]


def radec_to_vec(ra_deg, dec_deg):
    ra = np.radians(ra_deg)
    dec = np.radians(dec_deg)

    return np.array([
        np.cos(dec) * np.cos(ra),
        np.cos(dec) * np.sin(ra),
        np.sin(dec)
    ], dtype=np.float64)


def rotate_around_axis(v, axis, theta):
    axis = axis / np.linalg.norm(axis)
    c = np.cos(theta)
    s = np.sin(theta)

    return (
        v * c +
        np.cross(axis, v) * s +
        axis * np.dot(axis, v) * (1 - c)
    )


# =========================================================
# MAIN FUNCTION
# =========================================================
def generate_startracker_view(
    region_center_coords,
    image_size=2048,
    fov_deg=10,
    radius_deg_gaia=15.0,
    roll_deg=0.0,
    draw_constellations=False,
    return_plot=False,
    min_flux=1e-6,
    observer_position=np.array([0.0, 0.0, 0.0]),
):

    # -------------------------
    # Load stars (always needed)
    # -------------------------
    star_positions, fluxes = load_gaia_region(
        region_center=region_center_coords,
        radius_deg=radius_deg_gaia
    )

    if len(star_positions) == 0:
        raise RuntimeError("No stars loaded")

    # -------------------------
    # Camera frame
    # -------------------------
    boresight = radec_to_vec(*region_center_coords)
    print("Boresight within Generate Startracker View: ", boresight)
    up_ref = np.array([0.0, 0.0, 1.0])

    if np.abs(np.dot(up_ref, boresight)) > 0.9:
        up_ref = np.array([0.0, 1.0, 0.0])

    x_base = np.cross(up_ref, boresight)
    x_base /= np.linalg.norm(x_base)

    y_base = np.cross(boresight, x_base)
    y_base /= np.linalg.norm(y_base)

    z_base = boresight

    roll_rad = np.radians(roll_deg)

    x_cam = rotate_around_axis(x_base, z_base, roll_rad)
    y_cam = rotate_around_axis(y_base, z_base, roll_rad)

    R_cam = R.from_matrix(np.vstack([x_cam, y_cam, z_base]))

    # -------------------------
    # Transform + normalize
    # -------------------------
    rel = star_positions - observer_position
    stars_cam = R_cam.apply(rel)

    norms = np.linalg.norm(stars_cam, axis=1)
    stars_cam = stars_cam / norms[:, None]

    # -------------------------
    # Projection (always computed)
    # -------------------------
    fov = np.radians(fov_deg)
    scale = (image_size / 2.0) / np.tan(fov / 2.0)

    # Use the camera-frame vectors that were computed earlier
    x_all, y_all, z_all = stars_cam[:, 0], stars_cam[:, 1], stars_cam[:, 2]
    flux_all = fluxes

    # z>0 filter (in front of camera)
    mask_z = z_all > 0
    if not np.any(mask_z):
        # nothing visible
        px = np.empty((0,), dtype=np.int32)
        py = np.empty((0,), dtype=np.int32)
        f = np.empty((0,), dtype=np.float32)
        stars_cam_visible = np.empty((0, 3), dtype=np.float32)
    else:
        x = x_all[mask_z]
        y = y_all[mask_z]
        z = z_all[mask_z]
        f = flux_all[mask_z]
        stars_cam_visible = stars_cam[mask_z]

        u = x / z
        v = y / z

        px = (image_size / 2.0 + scale * u).astype(np.int32)
        py = (image_size / 2.0 + scale * v).astype(np.int32)

        inside = (
                (px >= 0) & (px < image_size) &
                (py >= 0) & (py < image_size)
        )

        px = px[inside]
        py = py[inside]
        f = f[inside]
        stars_cam_visible = stars_cam_visible[inside]

        bright = f > min_flux
        px = px[bright]
        py = py[bright]
        f = f[bright]
        stars_cam_visible = stars_cam_visible[bright]

    # =========================================================
    # ONLY BUILD IMAGE IF REQUESTED
    # =========================================================
    image = None
    dipper_pixels = None
    psf_kernel = psf_r = None

    if return_plot:
        psf_kernel, psf_r = build_psf(image_size)
        image = np.zeros((image_size, image_size), dtype=np.float32)

        for xpix, ypix, flux_val in zip(px, py, f):
            add_star_psf(image, xpix, ypix, flux_val, psf_kernel, psf_r)

        if draw_constellations:
            dipper_vecs = get_constellation_vectors(BIG_DIPPER_STARS)

            def project(vec):
                x, y, z = vec
                if z <= 0:
                    return None
                u = x / z
                v = y / z
                px_ = int(image_size / 2 + scale * u)
                py_ = int(image_size / 2 + scale * v)

                if 0 <= px_ < image_size and 0 <= py_ < image_size:
                    return (px_, py_)
                return None

            dipper_cam = {k: R_cam.apply(v) for k, v in dipper_vecs.items()}
            dipper_pixels = {k: project(v) for k, v in dipper_cam.items()}

    # -------------------------
    # Output (always lightweight)
    # -------------------------
    return {
        "pixel_x": px,
        "pixel_y": py,
        "flux": f,
        "camera_vectors": stars_cam_visible.astype(np.float32),
        "fov_deg": fov_deg,
        "scale": scale,
        "dipper_pixels": dipper_pixels,
        "image": image,  # will be None unless return_plot=True
        #"figure": figure,  # will be None unless return_plot=True
    }
