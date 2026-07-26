from Spacecraft.Vehicle import Vehicle
import matplotlib
from scipy.spatial.transform import Rotation as R
from matplotlib.animation import FuncAnimation
import math as mat
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import PillowWriter
from Basilisk.utilities import SimulationBaseClass, macros, vizSupport, RigidBodyKinematics
from Sensors.Sensors import SensorsManager
from Basilisk.simulation import mujoco, svIntegrators
from Basilisk.architecture import messaging
import os
from Dynamics.Dynamics import ConstantGravity
from VisualizationHelpers import ThrusterVizMessageWriter

matplotlib.use("TkAgg")


runtime = 400.0 #how long to run the simulation for, in seconds
samp = 0.1 #sampling rate, ie how often to take measurements, in seconds
sampling_ns = macros.sec2nano(samp)  # how often to sample sensors in ns
sim = SimulationBaseClass.SimBaseClass()                        # Initialize/instantiate a simulation environment

THRUSTER_NAME = "VR900"
SPACECRAFT_BODY_NAME = "IMX"
MJXML = "Spacecraft/spacecraft.xml"
OBJ_NAME = "Luna"
THRUSTER_LOCATION = [0.0, 0.0, -1.0] #Meters
THRUSTER_DIRECTION = [0.0, 0.0, 1.0] # [-] ?
THRUSTER_VIZ_SCALE = 10.0   # [-] ?
OBJ_VIZ_SCALE = 1000.0


def _get_body_geom_info(scene: mujoco.MJScene, body_name: str):
    """Return the first MuJoCo geom attached to a scene body."""
    geomInfos = scene.getGeomInfos()
    for geomIndex in range(len(geomInfos)):
        geomInfo = geomInfos[geomIndex]
        if geomInfo.bodyName == body_name:
            return geomInfo
    raise ValueError(f"Could not find a MuJoCo geom for body '{body_name}'.")


def _attach_thruster_visualization(viz, spacecraft_name: str, writer):
    """Attach a MuJoCo thruster visualization message to a Vizard body."""
    from Basilisk.simulation import vizInterface

    for scDataIndex in range(len(viz.scData)):
        scData = viz.scData[scDataIndex]
        if scData.spacecraftName != spacecraft_name:
            continue

        thrInfo = vizInterface.ThrClusterMap()
        thrInfo.thrTag = writer.ModelTag
        thrInfo.color = vizSupport.toRGBA255("red")
        scData.thrInMsgs = messaging.THROutputMsgInMsgsVector(
            [writer.thrOutMsg.addSubscriber()]
        )
        scData.thrInfo = vizInterface.ThrClusterVector([thrInfo])
        vizSupport.setActuatorGuiSetting(
            viz,
            spacecraftName=spacecraft_name,
            viewThrusterHUD=True,
        )
        return

    raise ValueError(
        f"Could not find spacecraft '{spacecraft_name}' in Vizard spacecraft data."
    )


# Create simulation tasks and processes
process = sim.CreateNewProcess("proc")                          # Create a new simulation process
task = sim.CreateNewTask("record", sampling_ns)      # Create a new task in the simulation
process.addTask(task)                                               # Add created task to the process

#Continued simulation modules setup
sc=Vehicle(sim, SPACECRAFT_BODY_NAME, sampling_ns)                                #initialize the vehicle
SM = SensorsManager(sc, sampling_ns)                                #initialize the sensors manager

sc.initialize_sensors(SM)                                   #attach sensors to vehicle



#Initialize MuJoCo scene
LUNAR_OBJ_PATH = os.path.abspath(
    os.path.join(
        "Environment",
        "Objects",
        "ItokawaHayabusa.obj"
    )
)

LUNAR_TEXTURE_PATH = os.path.abspath(
    os.path.join(
            "Environment",
            "Objects",
            "ItokawaGrayscale.jpg"
        )
)

scene = mujoco.MJScene.fromFile(MJXML,files=[LUNAR_OBJ_PATH])
sim.AddModelToTask("record", scene)

#Set the integrator of the scene
integ = svIntegrators.svIntegratorRKF45(scene)
integ.setRelativeTolerance(1e-3)
integ.setAbsoluteTolerance(1e-3)
scene.setIntegrator(integ)

#Gravity Model - THIS WILL CHANGE WHEN WE HAVE MORE DYNAMICS/EPHEM DATA FOR THE MOON
gravity = ConstantGravity(force_N=[0.0, 0.0, -200.0]) #In newtons
scene.AddModelToDynamicsTask(gravity)
gravityApplicationSite = scene.getBody(SPACECRAFT_BODY_NAME).getOrigin()

#Thruster - THIS WILL CHANGE WHEN WE HAVE THE ABILITY FOR THRUSTER CONTROL IN OTHER MODULES
gravityActuator: mujoco.MJForceActuator = scene.addForceActuator(
    "hub_gravity", gravityApplicationSite
)
gravityActuator.forceInMsg.subscribeTo(gravity.forceOutMsg)

gravity.frameInMsg.subscribeTo(gravityApplicationSite.stateOutMsg) #Change the forceactuator from the site-fixed reference frame to the inertial reference frame

thrust = 75.0 #Thruster strength in newtons
thrustMsg = messaging.SingleActuatorMsg()
thrustMsg.write(messaging.SingleActuatorMsgPayload(input=thrust))
scene.getSingleActuator(THRUSTER_NAME).actuatorInMsg.subscribeTo(thrustMsg)

sc.attach_scene(scene)                                          #For recorder to work properly

if vizSupport.vizFound:
    asteroidGeom = _get_body_geom_info(scene, OBJ_NAME)
    thrusterVizWriter = ThrusterVizMessageWriter(
        THRUSTER_NAME,
        thrustMsg,
        thrust,
        THRUSTER_LOCATION,
        THRUSTER_DIRECTION,
        THRUSTER_VIZ_SCALE,
    )
    sim.AddModelToTask("record", thrusterVizWriter)

    viz = vizSupport.enableUnityVisualization(
        sim,
        "record",
        scene,
        saveFile=__file__,
    )

    _attach_thruster_visualization(
        viz,
        SPACECRAFT_BODY_NAME,
        thrusterVizWriter,
    )

    viz.settings.showSpacecraftAsSprites = -1
    viz.settings.ambient = 0.1
    viz.settings.spacecraftShadowBrightness = 0.07
    vizSupport.createCustomModel(
        viz,
        modelPath=LUNAR_OBJ_PATH,
        simBodiesToModify=[OBJ_NAME],
        scale=[
            OBJ_VIZ_SCALE,
            OBJ_VIZ_SCALE,
            OBJ_VIZ_SCALE,
        ],
        offset=list(asteroidGeom.pos),
        rotation=list(RigidBodyKinematics.EP2Euler321(list(asteroidGeom.quat))),
        customTexturePath=LUNAR_TEXTURE_PATH,
        shader=1,
    )



# Run simulation
sim.InitializeSimulation()  # Start the simulation
sim.ConfigureStopTime((mat.floor(runtime / samp) * sampling_ns))  # When the simulation should stop
sim.ExecuteSimulation()

out = sc.output()

#vvvvvv everything else is just diagnostics and plotting below vvvvvv
##region plotting
for line in out:
    print(line)
    print()



true = out["true_data"]

position      = true[0]
velocity      = true[1]
attitude      = true[2]
angular_rate  = true[3]

from mpl_toolkits.mplot3d import Axes3D

fig = plt.figure(figsize=(8, 8))
ax = fig.add_subplot(111, projection='3d')

# Plot trajectory
ax.plot(position[:, 0], position[:, 1], position[:, 2], lw=2)

# Mark start and end points
ax.scatter(position[0, 0], position[0, 1], position[0, 2],
           color='green', s=50, label='Start')
ax.scatter(position[-1, 0], position[-1, 1], position[-1, 2],
           color='red', s=50, label='End')

ax.set_xlabel('X Position [m]')
ax.set_ylabel('Y Position [m]')
ax.set_zlabel('Z Position [m]')
ax.set_title('3D Spacecraft Trajectory')

# Make all axes use the same scale
x = position[:, 0]
y = position[:, 1]
z = position[:, 2]

max_range = np.array([
    x.max() - x.min(),
    y.max() - y.min(),
    z.max() - z.min()
]).max() / 2

mid_x = (x.max() + x.min()) / 2
mid_y = (y.max() + y.min()) / 2
mid_z = (z.max() + z.min()) / 2

ax.set_xlim(mid_x - max_range, mid_x + max_range)
ax.set_ylim(mid_y - max_range, mid_y + max_range)
ax.set_zlim(mid_z - max_range, mid_z + max_range)

# Floor (bottom of plot)
z_floor = z.min()

# Start point
x0, y0, z0 = position[0]

# End point
x1, y1, z1 = position[-1]
# Start point
ax.plot([x0, x0], [y0, y0], [z_floor, z0], 'g--', lw=1)     # vertical
ax.plot([x0, x0], [y.min(), y0], [z_floor, z_floor], 'g:', lw=1)
ax.plot([x.min(), x0], [y0, y0], [z_floor, z_floor], 'g:', lw=1)

# End point
ax.plot([x1, x1], [y1, y1], [z_floor, z1], 'r--', lw=1)
ax.plot([x1, x1], [y.min(), y1], [z_floor, z_floor], 'r:', lw=1)
ax.plot([x.min(), x1], [y1, y1], [z_floor, z_floor], 'r:', lw=1)

ax.legend()
plt.tight_layout()
plt.show()
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
    "AccelPlatform",
    "AngVelPlatform"
]
fig, axs = plt.subplots(2, 1, figsize=(14, 10), sharex=True)

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
    xR = R_i.T @ xB #transponse required because R_i maps inertial -> body
    yR = R_i.T @ yB
    zR = R_i.T @ zB

    # Plot rotated axes
    scale = 0.6
    ax.quiver(0,0,0, *(scale * xR), color='r', label='X axis')
    ax.quiver(0,0,0, *(scale * yR), color='g', label='Y axis')
    ax.quiver(0,0,0, *(scale * zR), color='b', label='Z axis')

    ax.legend()

ani = FuncAnimation(fig, update, frames=num_samples, interval=10)


# Save as GIF
writer = PillowWriter(fps=30)
#ani.save("attitude_animation.gif", writer=writer) #uncomment this in order to save gif of star tracker output

plt.show(block=True)

##endregion

