from Sensors.Sensors import SensorsManager
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import numpy as np


out = SensorsManager().run_single_timestep_test(500)

for line in out:
    print(out[line])
    print()



# Extract IMU and ST dictionaries
imu = out["imu"]["IMU"]
st  = out["star_tracker"]["ST"]

# Number of samples (3 samples for 2 timesteps: t=0, 0.01, 0.02)
num_samples = len(next(iter(imu.values())))
t = np.arange(num_samples) * 0.01   # 0.01 s timestep

# -----------------------------
# Plot IMU fields
# -----------------------------
imu_fields = [
    "DVFramePlatform",
    "AccelPlatform",
    "DRFramePlatform",
    "AngVelPlatform"
]

plt.figure(figsize=(12, 8))

for field in imu_fields:
    arr = imu[field]     # shape (N,3)
    for axis in range(3):
        plt.plot(t, arr[:, axis], label=f"{field}[{axis}]")

plt.title("IMU Logged Fields Over Time")
plt.xlabel("Time [s]")
plt.ylabel("IMU Values")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()

# -----------------------------
# Plot StarTracker quaternion
# -----------------------------
q = st["qInrtl2Case"]    # shape (N,4)

alpha = 0.05
q_filt = np.zeros_like(q)
q_filt[0] = q[0]

for i in range(1, len(q)):
    q_filt[i] = alpha*q[i] + (1-alpha)*q_filt[i-1]
    q_filt[i] /= np.linalg.norm(q_filt[i])


plt.figure(figsize=(12, 8))
for i in range(4):
    plt.plot(t, q_filt[:, i], label=f"qInrtl2Case[{i}]")

plt.title("StarTracker Quaternion Over Time")
plt.xlabel("Time [s]")
plt.ylabel("Quaternion Component")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()


from scipy.spatial.transform import Rotation as R
from matplotlib.animation import FuncAnimation

# Convert quaternions (Basilisk gives [q1,q2,q3,q0] = [x,y,z,w])
# SciPy expects [x,y,z,w], so this is already correct ordering.
R_mats = R.from_quat(q_filt).as_matrix()   # shape (N,3,3)

# Canonical body axes
xB = np.array([1,0,0])
yB = np.array([0,1,0])
zB = np.array([0,0,1])

fig = plt.figure(figsize=(8,8))
ax = fig.add_subplot(111, projection='3d')

def update(i):
    ax.cla()
    ax.set_xlim([-1,1])
    ax.set_ylim([-1,1])
    ax.set_zlim([-1,1])
    ax.set_title(f"Attitude at t={t[i]:.2f}s")

    R_i = R_mats[i]

    # Rotate body axes
    xR = R_i @ xB
    yR = R_i @ yB
    zR = R_i @ zB

    # Plot rotated axes
    scale = 0.2
    ax.quiver(0,0,0, *(scale * xR), color='r', label='X axis')
    ax.quiver(0,0,0, *(scale * yR), color='g', label='Y axis')
    ax.quiver(0,0,0, *(scale * zR), color='b', label='Z axis')

    ax.legend()

ani = FuncAnimation(fig, update, frames=num_samples, interval=50)
plt.show(block=True)