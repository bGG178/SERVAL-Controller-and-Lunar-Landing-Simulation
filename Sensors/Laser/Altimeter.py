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
        self.contact_stiffness = 1.0e5
        self.contact_damping = 2.0e4
        self.contact_tangential_damping = 5000.0
        self.contact_friction_coefficient = 0.8
        self.max_contact_force = 2.0e5
        self.use_radial_contact_normal = False
        self.apply_contact_torque = True
        self.max_contact_torque = 0.5e3
        self.settle_velocity_threshold = 0.02  # m/s
        self.settle_damping = 50000.0  # N/(m/s)  Damping used to remove small residual tangential motion.
        self.max_settle_force = 2.0e4  # Maximum force available to kill residual motion.

        # ==============================================================
        # CONTACT DEBUGGING
        # ==============================================================

        self.debug_contacts = True
        self.debug_contact_every_n_steps = 20
        self._debug_step_counter = 0

        self.terrain_geom_id = mujoco.mj_name2id(
            model,
            mujoco.mjtObj.mjOBJ_GEOM,
            "lunarTerrain"
        )

        print("\n========== MUJOCO TERRAIN DEBUG ==========")
        print("Terrain geom ID:", self.terrain_geom_id)

        if self.terrain_geom_id >= 0:

            terrain_mesh_id = model.geom_dataid[self.terrain_geom_id]

            print("Terrain geom type:", model.geom_type[self.terrain_geom_id])
            print("Terrain mesh ID:", terrain_mesh_id)

            if terrain_mesh_id >= 0:
                print(
                    "Terrain mesh name:",
                    mujoco.mj_id2name(
                        model,
                        mujoco.mjtObj.mjOBJ_MESH,
                        terrain_mesh_id
                    )
                )

                print(
                    "Terrain mesh vertices:",
                    model.mesh_vertnum[terrain_mesh_id]
                )

                print(
                    "Terrain mesh faces:",
                    model.mesh_facenum[terrain_mesh_id]
                )

        print("==========================================\n")

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
        omega_BN_N = C_BN @ omega_BN_B

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

        print("Contact force: ", F_N, "Torque: ", torque_B)

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

        self._debug_step_counter += 1

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

            # ----------------------------------------------------------
            # Only spacecraft <-> terrain
            # ----------------------------------------------------------

            spacecraft_contact = (
                    geom1 == self.spacecraft_geom_id
                    or
                    geom2 == self.spacecraft_geom_id
            )

            if not spacecraft_contact:
                continue

            self.collision = True

            normal = contact.frame[:3].copy()
            normal /= max(np.linalg.norm(normal), 1e-12)

            penetration = max(0.0, -contact.dist)

            self.contact_point_mjc = contact.pos.copy()
            self.contact_normal_mjc = normal
            self.penetration = penetration

            # ----------------------------------------------------------
            # DIAGNOSTICS
            # ----------------------------------------------------------

            if (
                    self.debug_contacts
                    and self._debug_step_counter % self.debug_contact_every_n_steps == 0
            ):

                print("\n" + "=" * 70)
                print("MUJOCO CONTACT")
                print("=" * 70)

                print("Contact index:", i)

                print(
                    "Geom 1:",
                    geom1,
                    name1
                )

                print(
                    "Geom 2:",
                    geom2,
                    name2
                )

                print("Contact distance:", contact.dist)
                print("Penetration:", penetration)

                print("\nContact position [MuJoCo]:")
                print(contact.pos)

                print("\nContact normal [MuJoCo]:")
                print(normal)

                print("\nFull contact frame:")
                print(contact.frame.reshape(3, 3))

                # ------------------------------------------------------
                # Spacecraft COM
                # ------------------------------------------------------

                qpos_adr = self.spacecraft_qpos_adr

                spacecraft_pos = self.data.qpos[
                    qpos_adr:qpos_adr + 3
                ]

                print("\nSpacecraft COM:")
                print(spacecraft_pos)

                # ------------------------------------------------------
                # Vector COM -> contact
                # ------------------------------------------------------

                r_contact = contact.pos - spacecraft_pos

                print("\nCOM -> contact:")
                print(r_contact)

                print(
                    "COM -> contact distance:",
                    np.linalg.norm(r_contact)
                )

                # ------------------------------------------------------
                # RADIAL NORMAL
                # ------------------------------------------------------

                radial = spacecraft_pos.copy()
                radial_norm = np.linalg.norm(radial)

                if radial_norm > 0.0:
                    radial /= radial_norm

                    print("\nRadial normal:")
                    print(radial)

                    print(
                        "Angle between radial and MuJoCo normal [deg]:",
                        np.degrees(
                            np.arccos(
                                np.clip(
                                    np.dot(radial, normal),
                                    -1.0,
                                    1.0
                                )
                            )
                        )
                    )

                # ------------------------------------------------------
                # Contact force BEFORE any radial-normal override
                # ------------------------------------------------------

                contact_force = np.zeros(6)

                mujoco.mj_contactForce(
                    self.model,
                    self.data,
                    i,
                    contact_force
                )

                print("\nMuJoCo contact force:")
                print(contact_force)

                print("\n" + "=" * 70)

        # --------------------------------------------------------------
        # IMPORTANT:
        #
        # DO NOT modify the MuJoCo contact normal here.
        # get_contact_loads() will use contact.frame[:3].
        # --------------------------------------------------------------

    def get_contact_loads(
            self,
            r_spacecraft_mjc,
            v_spacecraft_mjc,
            omega_spacecraft_mjc,
            C_BN,
            R_mjc_to_bsk):

        F_mjc = np.zeros(3)
        torque_B = np.zeros(3)

        for i in range(self.data.ncon):

            contact = self.data.contact[i]

            geom1 = contact.geom1
            geom2 = contact.geom2

            # ==============================================================
            # ONLY SPACECRAFT <-> TERRAIN CONTACTS
            # ==============================================================

            if (
                    geom1 != self.spacecraft_geom_id
                    and
                    geom2 != self.spacecraft_geom_id
            ):
                continue

            # ==============================================================
            # CONTACT NORMAL
            #
            # MuJoCo contact.frame[:3] points:
            #
            #       geom1 --> geom2
            #
            # We want the normal pointing:
            #
            #       terrain --> spacecraft
            #
            # because that is the direction of the reaction force on
            # the spacecraft.
            # ==============================================================

            normal_mjc = contact.frame[:3].copy()

            normal_norm = np.linalg.norm(normal_mjc)

            if normal_norm < 1e-12:
                continue

            normal_mjc /= normal_norm

            if geom1 == self.spacecraft_geom_id:
                # MuJoCo normal is spacecraft -> terrain
                # Reverse it so force points terrain -> spacecraft.
                normal_mjc *= -1.0

            elif geom2 == self.spacecraft_geom_id:
                # MuJoCo normal is terrain -> spacecraft.
                # Already correct.
                pass

            # ==============================================================
            # PENETRATION
            # ==============================================================

            penetration = max(0.0, -contact.dist)

            if penetration <= 0.0:
                continue

            # ==============================================================
            # CONTACT POINT RELATIVE TO SPACECRAFT COM
            # ==============================================================

            r_contact_from_com_mjc = (
                    contact.pos - r_spacecraft_mjc
            )

            # ==============================================================
            # CONTACT POINT VELOCITY
            # ==============================================================

            contact_velocity_mjc = (
                    v_spacecraft_mjc
                    +
                    np.cross(
                        omega_spacecraft_mjc,
                        r_contact_from_com_mjc
                    )
            )

            # ==============================================================
            # NORMAL VELOCITY
            #
            # Positive = spacecraft moving away from terrain
            # Negative = spacecraft moving into terrain
            # ==============================================================

            normal_velocity = np.dot(
                contact_velocity_mjc,
                normal_mjc
            )

            # ==============================================================
            # NORMAL SPRING-DAMPER FORCE
            # ==============================================================

            force_magnitude = (
                    self.contact_stiffness * penetration
                    -
                    self.contact_damping * normal_velocity
            )

            force_magnitude = max(
                0.0,
                force_magnitude
            )

            force_magnitude = min(
                force_magnitude,
                self.max_contact_force
            )

            force_mjc = (
                    force_magnitude * normal_mjc
            )

            print("\nCONTACT FORCE CONTRIBUTION")
            print("contact index:", i)
            print("geom1:", geom1)
            print("geom2:", geom2)
            print("dist:", contact.dist)
            print("penetration:", penetration)
            print("normal:", normal_mjc)
            print("normal velocity:", normal_velocity)
            print("force magnitude:", force_magnitude)
            print("force MJC:", force_mjc)

            # ==============================================================
            # TANGENTIAL FRICTION
            # ==============================================================

            tangential_velocity_mjc = (
                    contact_velocity_mjc
                    -
                    normal_velocity * normal_mjc
            )

            tangential_speed = np.linalg.norm(
                tangential_velocity_mjc
            )

            tangential_force_mjc = np.zeros(3)

            if tangential_speed > 1e-12:

                # ----------------------------------------------------------
                # Maximum available Coulomb friction
                # ----------------------------------------------------------

                friction_limit = (
                        self.contact_friction_coefficient
                        *
                        force_magnitude
                )

                # ----------------------------------------------------------
                # Viscous friction / damping
                #
                # This is what actually produces a friction force.
                # It opposes BOTH:
                #
                #   translational sliding
                #
                #   rotational motion at the contact point
                #
                # because tangential_velocity includes omega x r.
                # ----------------------------------------------------------

                tangential_force_mjc = (
                        -self.contact_tangential_damping
                        *
                        tangential_velocity_mjc
                )

                # ----------------------------------------------------------
                # Coulomb friction limit
                # ----------------------------------------------------------

                tangential_force_norm = np.linalg.norm(
                    tangential_force_mjc
                )

                if tangential_force_norm > friction_limit:
                    tangential_force_mjc *= (
                            friction_limit
                            /
                            tangential_force_norm
                    )

                # ----------------------------------------------------------
                # SETTLING
                #
                # Once the contact point is moving very slowly, use a
                # stronger damping force to eliminate residual creeping.
                # ----------------------------------------------------------

                if tangential_speed < self.settle_velocity_threshold:

                    settle_force = (
                            -self.settle_damping
                            *
                            tangential_velocity_mjc
                    )

                    settle_force_norm = np.linalg.norm(
                        settle_force
                    )

                    if settle_force_norm > friction_limit:
                        settle_force *= (
                                friction_limit
                                /
                                max(settle_force_norm, 1e-12)
                        )

                    tangential_force_mjc = settle_force

            force_mjc += tangential_force_mjc

            # ==============================================================
            # ACCUMULATE FORCE
            # ==============================================================

            F_mjc += force_mjc

            # ==============================================================
            # TORQUE
            # ==============================================================

            if self.apply_contact_torque:
                r_contact_from_com_N = (
                        R_mjc_to_bsk
                        @
                        r_contact_from_com_mjc
                )

                force_N = (
                        R_mjc_to_bsk
                        @
                        force_mjc
                )

                torque_N = np.cross(
                    r_contact_from_com_N,
                    force_N
                )

                torque_B += (
                        C_BN
                        @
                        torque_N
                )

            # ==============================================================
            # STORE DEBUG STATE
            # ==============================================================

            self.collision = True
            self.contact_force_mjc = force_mjc.copy()
            self.contact_point_mjc = contact.pos.copy()
            self.contact_normal_mjc = normal_mjc.copy()
            self.penetration = penetration

        # ==============================================================
        # TORQUE LIMIT
        # ==============================================================

        #print("\nTOTAL CONTACT FORCE")
        #print("F_mjc:", F_mjc)
        #print("F_N:", R_mjc_to_bsk @ F_mjc)
        #print("magnitude:", np.linalg.norm(F_mjc))

        torque_norm = np.linalg.norm(torque_B)

        if torque_norm > self.max_contact_torque:
            torque_B *= (
                    self.max_contact_torque
                    /
                    torque_norm
            )

        print("\nTORQUE DEBUG")

        print("torque before clamp:")
        print(torque_B)

        print("torque magnitude:")
        print(np.linalg.norm(torque_B))

        omega_B = C_BN @ omega_spacecraft_mjc

        print("omega_B:")
        print(omega_B)

        if np.linalg.norm(omega_B) > 1e-12:
            print(
                "torque · omega:",
                np.dot(torque_B, omega_B)
            )

        # ==============================================================
        # MUJOCO -> BASILISK
        # ==============================================================

        F_N = R_mjc_to_bsk @ F_mjc

        return F_N, torque_B

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


