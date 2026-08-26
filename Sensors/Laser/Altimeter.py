import Basilisk
import mujoco
import numpy as np
from Basilisk.architecture import messaging, sysModel, bskLogging
from Basilisk.utilities import RigidBodyKinematics
from Spacecraft.Mujoco.FrameTransforms import R_BSK_TO_MUJOCO, R_MUJOCO_TO_BSK

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
        self.spacecraft_geom_id = mujoco.mj_name2id(
            model,
            mujoco.mjtObj.mjOBJ_GEOM,
            "spacecraft_body"
        )

        # Basilisk input: spacecraft state
        self.scStateInMsg = messaging.SCStatesMsgReader()

        # Basilisk output: NavTransMsg
        self.sensorOutMsg = messaging.NavTransMsg()
        self.forceOutMsg = messaging.CmdForceBodyMsg() #For the contact collision

        # Sensor outputs
        self.altitude = np.nan
        self.hit_geom = -1

        # Fields for logging
        self.fields = ["timeTag", "r_BN_N", "v_BN_N"]

        self.recorder = None
        self.collision = False
        self.contact_force_mjc = np.zeros(3)
        self.contact_point_mjc = np.zeros(3)
        self.contact_normal_mjc = np.zeros(3)
        self.penetration = 0.0

        self.spacecraft_body_id = mujoco.mj_name2id(
            model,
            mujoco.mjtObj.mjOBJ_BODY,
            "spacecraft"
        )

        if self.spacecraft_geom_id < 0:
            raise RuntimeError("MuJoCo model is missing geom 'spacecraft_body'.")
        if self.spacecraft_body_id < 0:
            raise RuntimeError("MuJoCo model is missing body 'spacecraft'.")

        self.spacecraft_mocap_id = model.body_mocapid[
            self.spacecraft_body_id
        ]

        if self.spacecraft_mocap_id < 0:
            raise RuntimeError(
                "MuJoCo body 'spacecraft' must be mocap='true' because "
                "Basilisk is the spacecraft dynamics source."
            )


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

        # ==============================================================
        # 1. BASILISK → MUJOCO FRAME
        # ==============================================================

        R_bsk_to_mjc = R_BSK_TO_MUJOCO
        R_mjc_to_bsk = R_MUJOCO_TO_BSK

        # ==============================================================
        # 2. READ BASILISK STATE
        # ==============================================================

        scState = self.scStateInMsg()

        r_BN_N = np.array(scState.r_BN_N, dtype=float)
        v_BN_N = np.array(scState.v_BN_N, dtype=float)
        sigma_BN = np.array(scState.sigma_BN, dtype=float)

        # ==============================================================
        # 3. POSITION
        # ==============================================================

        r_mjc = R_bsk_to_mjc @ r_BN_N

        self.data.mocap_pos[
            self.spacecraft_mocap_id
        ] = r_mjc

        # ==============================================================
        # 4. ATTITUDE
        # ==============================================================

        C_BN = RigidBodyKinematics.MRP2C(sigma_BN)
        C_NB = C_BN.T

        C_mjc = (
                R_bsk_to_mjc
                @ C_NB
                @ R_bsk_to_mjc.T
        )

        q_mjc = RigidBodyKinematics.C2EP(C_mjc)

        self.data.mocap_quat[
            self.spacecraft_mocap_id
        ] = q_mjc

        # ==============================================================
        # 5. VELOCITY
        # ==============================================================

        v_mjc = R_bsk_to_mjc @ v_BN_N

        # ==============================================================
        # 6. UPDATE MUJOCO
        # ==============================================================

        mujoco.mj_forward(
            self.model,
            self.data
        )

        # ==============================================================
        # 7. CHECK CONTACT
        # ==============================================================

        self.check_collision()

        # ==============================================================
        # 8. GET MUJOCO CONTACT FORCE
        # ==============================================================

        F_mjc = self.get_mujoco_contact_force()

        # ==============================================================
        # 9. MUJOCO → BASILISK
        # ==============================================================

        F_bsk = R_mjc_to_bsk @ F_mjc

        # ==============================================================
        # 10. PUBLISH FORCE
        # ==============================================================

        force_payload = self.forceOutMsg.zeroMsgPayload

        force_payload.forceRequestBody = [
            F_bsk[0],
            F_bsk[1],
            F_bsk[2]
        ]

        self.forceOutMsg.write(
            force_payload,
            CurrentSimNanos,
            self.moduleID
        )

        # ==============================================================
        # 11. LASER ALTIMETER
        # ==============================================================

        self.altitude, self.hit_geom = self.laser_altimeter()

        payload = self.sensorOutMsg.zeroMsgPayload

        payload.timeTag = CurrentSimNanos

        payload.r_BN_N = [
            0.0,
            0.0,
            self.altitude
        ]

        payload.v_BN_N = [
            0.0,
            0.0,
            0.0
        ]

        self.sensorOutMsg.write(
            payload,
            CurrentSimNanos,
            self.moduleID
        )
    # ======================================================================
    # MuJoCo raycast
    # ======================================================================
    def laser_altimeter(self):

        origin = self.data.site_xpos[self.laser_id].copy()
        rotation = self.data.site_xmat[self.laser_id].reshape(3, 3)

        direction = rotation[:, 2].copy()
        direction /= np.linalg.norm(direction)

        local_down = -origin
        local_down /= np.linalg.norm(local_down)

        if np.dot(direction, local_down) <= 0.0:
            return -1.0, -1

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

    def check_collision(self):

        self.collision = False
        self.contact_force_mjc[:] = 0.0
        self.contact_point_mjc[:] = 0.0
        self.contact_normal_mjc[:] = 0.0
        self.penetration = 0.0

        for i in range(self.data.ncon):

            contact = self.data.contact[i]

            geom1 = contact.geom1
            geom2 = contact.geom2

            name1 = mujoco.mj_id2name(
                self.model,
                mujoco.mjtObj.mjOBJ_GEOM,
                geom1
            )

            name2 = mujoco.mj_id2name(
                self.model,
                mujoco.mjtObj.mjOBJ_GEOM,
                geom2
            )

            # We only care about spacecraft ↔ terrain
            spacecraft_contact = (
                    name1 == "spacecraft_body"
                    or name2 == "spacecraft_body"
            )

            if not spacecraft_contact:
                continue

            self.collision = True

            self.contact_point_mjc = contact.pos.copy()

            # Contact frame's first axis is the contact normal
            self.contact_normal_mjc = contact.frame[:3].copy()

            # Negative distance means penetration
            self.penetration = max(0.0, -contact.dist)

            break

    def get_mujoco_contact_force(self):

        F_mjc = np.zeros(3)

        # No contacts
        if self.data.ncon == 0:
            return F_mjc

        for i in range(self.data.ncon):

            contact = self.data.contact[i]

            # Get the two geoms involved
            geom1 = contact.geom1
            geom2 = contact.geom2

            # Only consider contacts involving the spacecraft
            if (
                    geom1 != self.spacecraft_geom_id
                    and
                    geom2 != self.spacecraft_geom_id
            ):
                continue

            # MuJoCo contact force
            contact_force = np.zeros(6)

            mujoco.mj_contactForce(
                self.model,
                self.data,
                i,
                contact_force
            )

            # First 3 values are force in the contact frame
            force_contact = contact_force[:3]

            # contact.frame is a 3x3 rotation matrix stored flat
            frame = contact.frame.reshape(3, 3)

            # Convert contact-frame force → MuJoCo world frame
            force_world = frame.T @ force_contact

            # Make sure the force points ON the spacecraft
            if geom2 == self.spacecraft_geom_id:
                force_world *= -1.0

            F_mjc += force_world

        return F_mjc
