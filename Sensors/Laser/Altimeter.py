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

        # Basilisk outputs
        self.sensorOutMsg = messaging.NavTransMsg()
        self.forceOutMsg = messaging.CmdForceInertialMsg()
        self.torqueOutMsg = messaging.CmdTorqueBodyMsg()

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
        self.contact_margin = 0.25
        self.contact_stiffness = 5.0e5
        self.contact_damping = 8.0e4
        self.contact_tangential_damping = 0.0
        self.contact_friction_coefficient = 0.8
        self.max_contact_force = 5.0e5
        self.use_radial_contact_normal = True
        self.apply_contact_torque = False
        self.max_contact_torque = 2.0e3

        self.spacecraft_body_id = mujoco.mj_name2id(
            model,
            mujoco.mjtObj.mjOBJ_BODY,
            "spacecraft"
        )

        if self.spacecraft_geom_id < 0:
            raise RuntimeError("MuJoCo model is missing geom 'spacecraft_body'.")
        if self.spacecraft_body_id < 0:
            raise RuntimeError("MuJoCo model is missing body 'spacecraft'.")

        self.spacecraft_joint_id = mujoco.mj_name2id(
            model,
            mujoco.mjtObj.mjOBJ_JOINT,
            "spacecraft_freejoint"
        )
        if self.spacecraft_joint_id < 0:
            raise RuntimeError(
                "MuJoCo model is missing joint 'spacecraft_freejoint'."
            )

        self.spacecraft_qpos_adr = model.jnt_qposadr[
            self.spacecraft_joint_id
        ]
        self.spacecraft_qvel_adr = model.jnt_dofadr[
            self.spacecraft_joint_id
        ]


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

        self.publish_contact_loads(
            np.zeros(3),
            np.zeros(3),
            CurrentSimNanos
        )

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
        omega_BN_B = np.array(scState.omega_BN_B, dtype=float)

        # ==============================================================
        # 3. POSITION
        # ==============================================================

        r_mjc = R_bsk_to_mjc @ r_BN_N

        # ==============================================================
        # 4. ATTITUDE
        # ==============================================================

        C_BN = RigidBodyKinematics.MRP2C(sigma_BN)
        C_NB = C_BN.T
        omega_BN_N = C_NB @ omega_BN_B

        C_mjc = (
                R_bsk_to_mjc
                @ C_NB
                @ R_bsk_to_mjc.T
        )

        q_mjc = RigidBodyKinematics.C2EP(C_mjc)

        # ==============================================================
        # 5. VELOCITY
        # ==============================================================

        v_mjc = R_bsk_to_mjc @ v_BN_N
        omega_mjc = R_bsk_to_mjc @ omega_BN_N

        qpos_adr = self.spacecraft_qpos_adr
        qvel_adr = self.spacecraft_qvel_adr

        self.data.qpos[qpos_adr:qpos_adr + 3] = r_mjc
        self.data.qpos[qpos_adr + 3:qpos_adr + 7] = q_mjc
        self.data.qvel[qvel_adr:qvel_adr + 3] = v_mjc
        self.data.qvel[qvel_adr + 3:qvel_adr + 6] = omega_mjc

        mujoco.mj_normalizeQuat(
            self.model,
            self.data.qpos
        )

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
        # 8. COMPUTE CONTACT LOADS
        # ==============================================================

        F_N, torque_B = self.get_contact_loads(
            r_mjc,
            v_mjc,
            omega_mjc,
            C_BN,
            R_mjc_to_bsk
        )

        # ==============================================================
        # 9. MUJOCO → BASILISK
        # ==============================================================

        self.publish_contact_loads(
            F_N,
            torque_B,
            CurrentSimNanos
        )

        # ==============================================================
        # 10. LASER ALTIMETER
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

    def get_contact_loads(self, r_spacecraft_mjc, v_spacecraft_mjc,
                          omega_spacecraft_mjc,
                          C_BN, R_mjc_to_bsk):

        F_mjc = np.zeros(3)
        torque_B = np.zeros(3)

        for i in range(self.data.ncon):

            contact = self.data.contact[i]
            geom1 = contact.geom1
            geom2 = contact.geom2

            if (
                    geom1 != self.spacecraft_geom_id
                    and
                    geom2 != self.spacecraft_geom_id
            ):
                continue

            normal_mjc = contact.frame[:3].copy()
            normal_norm = np.linalg.norm(normal_mjc)
            if normal_norm <= 0.0:
                continue

            normal_mjc /= normal_norm
            if self.use_radial_contact_normal:
                radial_norm = np.linalg.norm(r_spacecraft_mjc)
                if radial_norm > 0.0:
                    normal_mjc = r_spacecraft_mjc / radial_norm

            contact_to_spacecraft = r_spacecraft_mjc - contact.pos
            if np.dot(normal_mjc, contact_to_spacecraft) < 0.0:
                normal_mjc *= -1.0

            r_contact_from_com_mjc = contact.pos - r_spacecraft_mjc
            contact_velocity_mjc = (
                    v_spacecraft_mjc
                    + np.cross(omega_spacecraft_mjc, r_contact_from_com_mjc)
            )
            normal_velocity = np.dot(contact_velocity_mjc, normal_mjc)
            penetration = max(0.0, -contact.dist)
            force_magnitude = max(
                0.0,
                self.contact_stiffness * penetration
                - self.contact_damping * normal_velocity
            )
            force_magnitude = min(force_magnitude, self.max_contact_force)

            force_mjc = force_magnitude * normal_mjc

            tangential_velocity_mjc = (
                    contact_velocity_mjc
                    - normal_velocity * normal_mjc
            )
            tangential_speed = np.linalg.norm(tangential_velocity_mjc)
            if tangential_speed > 0.0 and force_magnitude > 0.0:
                tangential_force_mjc = (
                        -self.contact_tangential_damping
                        * tangential_velocity_mjc
                )
                tangential_force_limit = (
                        self.contact_friction_coefficient
                        * force_magnitude
                )
                tangential_force_norm = np.linalg.norm(tangential_force_mjc)
                if tangential_force_norm > tangential_force_limit:
                    tangential_force_mjc *= (
                            tangential_force_limit
                            / tangential_force_norm
                    )
                force_mjc += tangential_force_mjc

            F_mjc += force_mjc

            if self.apply_contact_torque:
                r_contact_from_com_N = R_mjc_to_bsk @ r_contact_from_com_mjc
                force_N = R_mjc_to_bsk @ force_mjc
                torque_B += C_BN @ np.cross(r_contact_from_com_N, force_N)

            self.collision = True
            self.contact_force_mjc = force_mjc
            self.contact_point_mjc = contact.pos.copy()
            self.contact_normal_mjc = normal_mjc
            self.penetration = penetration

        torque_norm = np.linalg.norm(torque_B)
        if torque_norm > self.max_contact_torque:
            torque_B *= self.max_contact_torque / torque_norm

        return R_mjc_to_bsk @ F_mjc, torque_B

    def publish_contact_loads(self, force_N, torque_B, CurrentSimNanos):

        force_payload = self.forceOutMsg.zeroMsgPayload
        force_payload.forceRequestInertial = [
            force_N[0],
            force_N[1],
            force_N[2]
        ]
        self.forceOutMsg.write(
            force_payload,
            CurrentSimNanos,
            self.moduleID
        )

        torque_payload = self.torqueOutMsg.zeroMsgPayload
        torque_payload.torqueRequestBody = [
            torque_B[0],
            torque_B[1],
            torque_B[2]
        ]
        self.torqueOutMsg.write(
            torque_payload,
            CurrentSimNanos,
            self.moduleID
        )

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
