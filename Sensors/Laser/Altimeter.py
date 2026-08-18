import Basilisk
import mujoco
import numpy as np
from Basilisk.architecture import messaging, sysModel, bskLogging
from Basilisk.utilities import RigidBodyKinematics

class LaserAltimeter(sysModel.SysModel):
    """
    MuJoCo laser altimeter implemented as a Basilisk Python module.
    Basilisk controls spacecraft dynamics; MuJoCo provides geometry
    and raycasting for the laser measurement.
    """

    def __init__(self, model, data, laser_id):
        super().__init__()

        self.name = "LaserAltimeter1"

        self.ModelTag = "LaserAltimeter"
        self.model = model
        self.data = data
        self.laser_id = laser_id

        # Basilisk input: spacecraft state
        self.scStateInMsg = messaging.SCStatesMsgReader()

        # Basilisk output: NavTransMsg
        self.sensorOutMsg = messaging.NavTransMsg()

        # Sensor outputs
        self.altitude = np.nan
        self.hit_geom = -1

        # Fields for logging
        self.fields = ["timeTag", "r_BN_N", "v_BN_N"]

        self.recorder = None

    # ======================================================================
    # Reset() — called once when the module is added to the simulation
    # ======================================================================
    def Reset(self, CurrentSimNanos):

        if not self.scStateInMsg.isLinked():
            self.bskLogger.bskLog(
                bskLogging.BSK_ERROR,
                "LaserAltimeter.scStateInMsg is not linked."
            )

        payload = self.sensorOutMsg.zeroMsgPayload
        payload.timeTag = CurrentSimNanos
        payload.r_BN_N = [0.0, 0.0, 0.0]
        payload.v_BN_N = [0.0, 0.0, 0.0]

        self.sensorOutMsg.write(payload, CurrentSimNanos, self.moduleID)

        self.bskLogger.bskLog(
            bskLogging.BSK_INFORMATION,
            "LaserAltimeter Reset() complete."
        )

    # ======================================================================
    # UpdateState() — called every simulation step
    # ======================================================================
    def UpdateState(self, CurrentSimNanos):

        # --------------------------------------------------
        # CONSTANTS
        # --------------------------------------------------

        # Basilisk → MuJoCo rotation matrix
        # Basilisk gravity = +Y
        # MuJoCo gravity = -Z
        R_bsk_to_mjc = np.array([
            [1, 0, 0],  # X stays X
            [0, 0, 1],  # Z_bsk → Y_mjc
            [0, 1, 0]  # Y_bsk → Z_mjc
        ])

        # --------------------------------------------------
        # 1. Read spacecraft state from Basilisk
        # --------------------------------------------------
        scState = self.scStateInMsg()
        r_BN_N = np.array(scState.r_BN_N)  # Basilisk inertial position
        sigma_BN = np.array(scState.sigma_BN)  # Basilisk attitude (MRP)



        # --------------------------------------------------
        # 2. Convert Basilisk attitude → MuJoCo quaternion
        # --------------------------------------------------
        C_bsk = RigidBodyKinematics.MRP2C(sigma_BN)
        C_mjc = R_bsk_to_mjc @ C_bsk @ R_bsk_to_mjc.T
        q_mjc = RigidBodyKinematics.C2EP(C_mjc)

        # --------------------------------------------------
        # 3. Convert Basilisk position → MuJoCo world frame
        # --------------------------------------------------
        # Basilisk altitude is in Y
        alt_bsk = r_BN_N[1]

        # Rotate Basilisk inertial → MuJoCo world
        r_mjc = R_bsk_to_mjc @ r_BN_N

        # Compute altitude from rotated Z
        alt_mjc = r_mjc[2]

        # Write MuJoCo world position
        self.data.qpos[0] = r_mjc[0]
        self.data.qpos[1] = r_mjc[1]
        self.data.qpos[2] = alt_mjc

        self.data.qpos[3:7] = q_mjc



        # --------------------------------------------------
        # 4. Forward MuJoCo state
        # --------------------------------------------------
        mujoco.mj_forward(self.model, self.data)

        # --------------------------------------------------
        # 5. Perform laser raycast
        # --------------------------------------------------
        self.altitude, self.hit_geom = self.laser_altimeter()

        # --------------------------------------------------
        # 6. Publish Basilisk message
        # --------------------------------------------------
        payload = self.sensorOutMsg.zeroMsgPayload
        payload.timeTag = CurrentSimNanos
        payload.r_BN_N = [0.0, 0.0, self.altitude]
        payload.v_BN_N = [0.0, 0.0, 0.0]

        self.sensorOutMsg.write(payload, CurrentSimNanos, self.moduleID)

        # --------------------------------------------------
        # 7. Debug print
        # --------------------------------------------------
        if self.hit_geom == -1:
            pass
            #print(f"t={CurrentSimNanos * 1e-9:.3f}s | Laser: NO RETURN\n")
        else:
            print(
                f"t={CurrentSimNanos * 1e-9:.3f}s | "
                f"Altitude={self.altitude:.3f} m | Hit geom={self.hit_geom}"

            )
            # Debug
            #print(
            #    f"t={CurrentSimNanos * 1e-9:.3f} | "
            #    f"r = {scState.r_BN_N} | "
            #    f"v = {scState.v_BN_N}"
            #)
            print(f"Basilisk Y (alt) = {r_BN_N[1]:.3f} | MuJoCo Z = {self.data.qpos[2]:.3f}")
            print(f"Ray origin Z = {self.data.site_xpos[self.laser_id][2]:.3f}\n")

    # ======================================================================
    # MuJoCo raycast
    # ======================================================================
    def laser_altimeter(self):

        origin = self.data.site_xpos[self.laser_id].copy()
        rotation = self.data.site_xmat[self.laser_id].reshape(3, 3)

        direction = rotation[:, 2].copy()
        direction /= np.linalg.norm(direction)

        geomgroup = np.ones(6, dtype=np.uint8)
        geom_id = np.array([-1], dtype=np.int32)

        spacecraft_body_id = mujoco.mj_name2id(
            self.model,
            mujoco.mjtObj.mjOBJ_BODY,
            "spacecraft"
        )

        distance = mujoco.mj_ray(
            self.model,
            self.data,
            origin,
            direction,
            geomgroup,
            1,
            spacecraft_body_id,
            geom_id
        )

        return distance, geom_id[0]