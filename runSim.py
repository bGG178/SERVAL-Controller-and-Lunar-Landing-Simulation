from Spacecraft.Vehicle import Vehicle
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial.transform import Rotation as R
from matplotlib.animation import FuncAnimation

runtime = 5 #how long to run the simulation for, in seconds
samp = 0.01 #sampling rate, ie how often to take measurements, in seconds
sc=Vehicle(samp)
out = sc.run(runtime / samp)




#vvvvvv everything else is just diagnostics and plotting below vvvvvv

for line in out:
    print(out[line])
    print()



# Extract IMU and ST dictionaries
imu = out["imu"]["IMU"]
st  = out["star_tracker"]["ST"]

# Number of samples (3 samples for 2 timesteps: t=0, 0.01, 0.02)
num_samples = len(next(iter(imu.values())))
t = np.arange(num_samples) * samp

# -----------------------------
# Plot IMU fields
# -----------------------------
imu_fields = [
    "DVFramePlatform",
    "AccelPlatform",
    "DRFramePlatform",
    "AngVelPlatform"
]
fig, axs = plt.subplots(2, 2, figsize=(14, 10), sharex=True)

axs = axs.flatten()

for ax, field in zip(axs, imu_fields):
    arr = imu[field]   # shape (N, 3)

    ax.plot(t, arr[:, 0], label="X")
    ax.plot(t, arr[:, 1], label="Y")
    ax.plot(t, arr[:, 2], label="Z")

    ax.set_title(field)
    ax.set_xlabel("Time [s]")
    ax.set_ylabel(field)
    ax.grid(True)
    ax.legend()

plt.tight_layout()
plt.show()

# -----------------------------
# Plot StarTracker quaternion
# -----------------------------
q = st["qInrtl2Case"]    # shape (N,4)

#FILTERING BELOW! May not be akin to space-based filtering, was just first attempt
#alpha = 0.05
#q_filt = np.zeros_like(q)
#q_filt[0] = q[0]

#for i in range(1, len(q)):
#    q_filt[i] = alpha*q[i] + (1-alpha)*q_filt[i-1]
#    q_filt[i] /= np.linalg.norm(q_filt[i])
#q=q_filt

plt.figure(figsize=(12, 8))
for i in range(4):
    plt.plot(t, q[:, i+0], label=f"qInrtl2Case[{i+0}]")

plt.title("StarTracker Quaternion Over Time")
plt.xlabel("Time [s]")
plt.ylabel("Quaternion Component")
plt.grid(True)

plt.legend()
plt.tight_layout()
plt.show()



# Convert quaternions (Basilisk gives [q1,q2,q3,q0] = [x,y,z,w])
# SciPy expects [x,y,z,w], so this is already correct ordering.
R_mats = R.from_quat(q).as_matrix()   # shape (N,3,3)

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
    scale = 0.6
    ax.quiver(0,0,0, *(scale * xR), color='r', label='X axis')
    ax.quiver(0,0,0, *(scale * yR), color='g', label='Y axis')
    ax.quiver(0,0,0, *(scale * zR), color='b', label='Z axis')

    ax.legend()

ani = FuncAnimation(fig, update, frames=num_samples, interval=100)
plt.show(block=True)