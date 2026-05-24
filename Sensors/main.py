from StarTracker.star_tracker_sim import generate_startracker_view
from StarTracker.Startracker_processing import process_star_tracker_output
import numpy as np

data = generate_startracker_view(
    return_plot=False,
    region_center_coords=(180.0, 56.0)
)

result = process_star_tracker_output(
    data,
    region_center=(180.0, 56.0)
)
q = result["quaternion"]

print("\n=== ATTITUDE SOLUTION ===")
print(f"Quaternion (x, y, z, w):")
print(f"  [{q[0]: .8f}, {q[1]: .8f}, {q[2]: .8f}, {q[3]: .8f}]")
print(f"Norm: {np.linalg.norm(q):.6f}")
print(f"Matched stars: {result['matched_stars']}")
print(f"RMS error: {result['rms_error']:.6e}")

import numpy as np

from scipy.spatial.transform import Rotation as R

q = result["quaternion"]
rot = R.from_quat(q)

# camera boresight (forward direction)
forward = np.array([0, 0, 1])

sky_direction = rot.apply(forward)

print(sky_direction)
x, y, z = sky_direction

ra = np.degrees(np.arctan2(y, x)) % 360
dec = np.degrees(np.arcsin(z))

print("RA:", ra)
print("DEC:", dec)