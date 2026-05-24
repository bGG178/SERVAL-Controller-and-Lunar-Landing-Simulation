import numpy as np
from pathlib import Path
from astroquery.gaia import Gaia
from concurrent.futures import ThreadPoolExecutor

# =========================================================
# PATHS
# =========================================================

BASE_DIR = Path(__file__).resolve().parent

CACHE_DIR = BASE_DIR / "Starcache"
CACHE_DIR.mkdir(exist_ok=True)

TRI_CACHE_DIR = BASE_DIR / "TriStarCache"
TRI_CACHE_DIR.mkdir(exist_ok=True)

# =========================================================
# CONSTANTS
# =========================================================

TILE_SIZE = 2.0
TILE_PADDING = 0.25

# =========================================================
# MEMORY CACHE
# =========================================================

_MEMORY_TILE_CACHE = {}

# =========================================================
# HELPERS
# =========================================================

def tile_key(ra_tile, dec_tile):
    return int(ra_tile), int(dec_tile)


def tile_filename(ra_tile, dec_tile):
    return CACHE_DIR / f"tile_RA{ra_tile}_DEC{dec_tile}.npz"


def tri_cache_filename(ra_tile, dec_tile):
    return TRI_CACHE_DIR / f"tri_RA{ra_tile}_DEC{dec_tile}.npz"


def sky_tile(ra_deg, dec_deg):
    ra_deg = float(ra_deg) % 360.0
    dec_deg = float(dec_deg)

    return (
        int(np.floor(ra_deg / TILE_SIZE)),
        int(np.floor((dec_deg + 90.0) / TILE_SIZE))
    )

# =========================================================
# TRIANGLE SIGNATURE
# =========================================================

def triangle_signature(a, b, c):
    def ang(x, y):
        return np.arccos(np.clip(np.dot(x, y), -1.0, 1.0))

    e = np.array([
        ang(a, b),
        ang(b, c),
        ang(c, a)
    ], dtype=np.float32)

    return np.sort(e)  # ALWAYS (3,)

# =========================================================
# TILE COVERAGE
# =========================================================

def required_tiles(region_center, radius_deg):
    ra0, dec0 = region_center

    ra_min = (ra0 - radius_deg) % 360.0
    ra_max = (ra0 + radius_deg) % 360.0

    dec_min = max(dec0 - radius_deg, -90.0)
    dec_max = min(dec0 + radius_deg, 90.0)

    if ra_min < ra_max:
        ra_tiles = range(
            int(np.floor(ra_min / TILE_SIZE)),
            int(np.floor(ra_max / TILE_SIZE)) + 1
        )
    else:
        ra_tiles = list(range(int(np.floor(ra_min / TILE_SIZE)), int(360 / TILE_SIZE)))
        ra_tiles += list(range(0, int(np.floor(ra_max / TILE_SIZE)) + 1))

    dec_tiles = range(
        int(np.floor((dec_min + 90.0) / TILE_SIZE)),
        int(np.floor((dec_max + 90.0) / TILE_SIZE)) + 1
    )

    return [(r, d) for r in ra_tiles for d in dec_tiles]

# =========================================================
# TILE QUERY
# =========================================================

def query_tile(ra_tile, dec_tile):

    key = tile_key(ra_tile, dec_tile)

    if key in _MEMORY_TILE_CACHE:
        return _MEMORY_TILE_CACHE[key]

    cache_file = tile_filename(*key)

    if cache_file.exists():
        data = np.load(cache_file)
        result = (data["star_positions"], data["fluxes"])
        _MEMORY_TILE_CACHE[key] = result
        return result

    ra_min = ra_tile * TILE_SIZE - TILE_PADDING
    ra_max = ra_tile * TILE_SIZE + TILE_SIZE + TILE_PADDING

    dec_min = (dec_tile * TILE_SIZE - 90.0) - TILE_PADDING
    dec_max = (dec_tile * TILE_SIZE - 90.0) + TILE_SIZE + TILE_PADDING

    center_ra = (ra_min + ra_max) * 0.5
    center_dec = (dec_min + dec_max) * 0.5
    radius = max(ra_max - ra_min, dec_max - dec_min) * 0.75

    query = f"""
    SELECT ra, dec, parallax, phot_g_mean_mag
    FROM gaiadr3.gaia_source
    WHERE CONTAINS(
        POINT('ICRS', ra, dec),
        CIRCLE('ICRS', {center_ra}, {center_dec}, {radius})
    ) = 1
    AND phot_g_mean_mag < 21
    AND parallax > 0
    """

    job = Gaia.launch_job_async(query)
    stars = job.get_results()

    if len(stars) == 0:
        empty = (np.empty((0, 3)), np.empty((0,)))
        np.savez(cache_file, star_positions=empty[0], fluxes=empty[1])
        _MEMORY_TILE_CACHE[key] = empty
        return empty

    positions = []
    fluxes = []

    for r in stars:
        ra, dec = r["ra"], r["dec"]
        p, mag = r["parallax"], r["phot_g_mean_mag"]

        if np.isnan(ra) or np.isnan(dec) or np.isnan(p) or np.isnan(mag):
            continue

        dist = 1000.0 / p
        if dist > 10000:
            continue

        x = np.cos(np.radians(dec)) * np.cos(np.radians(ra))
        y = np.cos(np.radians(dec)) * np.sin(np.radians(ra))
        z = np.sin(np.radians(dec))

        positions.append(np.array([x, y, z]) * dist)
        fluxes.append(10 ** (-0.4 * mag))

    positions = np.array(positions, dtype=np.float64)
    fluxes = np.array(fluxes, dtype=np.float32)

    if len(fluxes) > 0:
        keep = np.argsort(fluxes)[int(len(fluxes) * 0.9):]
        positions = positions[keep]
        fluxes = fluxes[keep]

    np.savez(cache_file, star_positions=positions, fluxes=fluxes)
    _MEMORY_TILE_CACHE[key] = (positions, fluxes)

    build_triangle_cache(ra_tile, dec_tile)

    return positions, fluxes

# =========================================================
# TRIANGLE CACHE
# =========================================================

def build_triangle_cache(ra_tile, dec_tile, top_k=32):

    f = tri_cache_filename(ra_tile, dec_tile)
    if f.exists():
        return

    pos, flux = query_tile(ra_tile, dec_tile)
    if len(pos) < 3:
        return

    vecs = pos / np.linalg.norm(pos, axis=1, keepdims=True)

    idx = np.argsort(flux)[::-1][:top_k]
    vecs = vecs[idx]

    hashes = []
    ids = []

    n = len(vecs)

    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):

                hashes.append(triangle_signature(vecs[i], vecs[j], vecs[k]))
                ids.append([i, j, k])

    if not hashes:
        return

    hashes = np.stack(hashes).astype(np.float32)
    ids = np.array(ids, dtype=np.int32)

    np.savez(f, hashes=hashes, triangle_ids=ids, star_vectors=vecs, fluxes=flux[idx])

# =========================================================
# REGION LOADER
# =========================================================

def load_gaia_region(region_center=(10, 41), radius_deg=3):

    tiles = required_tiles(region_center, radius_deg)

    def load(t):
        return query_tile(*t)

    with ThreadPoolExecutor(max_workers=8) as ex:
        results = list(ex.map(load, tiles))

    pos, flux = [], []

    for p, f in results:
        if len(p) > 0:
            pos.append(p)
            flux.append(f)

    if not pos:
        return np.empty((0, 3)), np.empty((0,))

    return np.concatenate(pos), np.concatenate(flux)

# =========================================================
# TRIANGLE REGION LOADER
# =========================================================

def load_triangle_region(region_center=(10, 41), radius_deg=3):
    tiles = required_tiles(region_center, radius_deg)

    hashes_all = []
    ids_all = []

    for r, d in tiles:
        file = tri_cache_filename(r, d)

        if not file.exists():
            build_triangle_cache(r, d)

        if not file.exists():
            continue

        data = np.load(file)

        # ✅ FORCE SHAPE CONSISTENCY HERE
        hashes_all.append(np.asarray(data["hashes"]).reshape(-1, 3))
        ids_all.append(np.asarray(data["triangle_ids"]))

    if not hashes_all:
        return np.empty((0, 3), dtype=np.float32), np.empty((0, 3), dtype=np.int32)

    hashes = np.concatenate(hashes_all, axis=0)
    ids = np.concatenate(ids_all, axis=0)

    # safety check
    assert hashes.ndim == 2 and hashes.shape[1] == 3, hashes.shape

    return hashes, ids