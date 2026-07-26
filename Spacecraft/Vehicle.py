from Basilisk.utilities import SimulationBaseClass, macros, vizSupport
from Basilisk.simulation import spacecraft
from Sensors.Sensors import SensorsManager

class Vehicle:
    def __init__(self, sim, name, sampling_ns=0.01):
        self.imus = {}              # sensor IMU initialization, if you have multiple IMUs they get stored here
        self.star_trackers = {}     # sensor Star Tracker initialization, if you have multiple IMUs they get stored here
        self.lander = spacecraft.Spacecraft()        # Create a spacecraft instance from the spacecraft basilisk module
        self.lander.ModelTag = name             # Tag the spacecraft with a name

        self.lander.hub.r_CN_NInit = [70000, 0.0, 0.0]  # Current position within the inertial frame (in m)
        self.lander.hub.v_CN_NInit = [23500.0, 5000.0, 1000.0]  # Current velocity within the inertial frame (m/s)
        self.lander.hub.sigma_BNInit = [[0.0], [0.7], [0.0]]  # Current attitude with respect to the body and inertial frame utilizing a Modified Rodrigues Parameter (MRP) vector. (N->P)
        self.lander.hub.omega_BN_BInit = [[0.0], [0.0], [1.0]]  # Current angular velocity with body frame relative to inertial frame (rad/s)
        self.lander.hub.mHub = 2120.0                     #mass in kg
        self.lander.hub.r_BcB_B = [0,0,0]                   #Center of mass in B frame

        self.sim = sim
        self.SM = None                          #make none because it will be applied later


        self.loggers = {}                                   #Holds the logger instances


        self.sim.AddModelToTask("record", self.lander)  # Add spacecraft to task




    def initialize_sensors(self, SM):
        self.SM = SM

        # Add sensors to spacecraft
        self.SM.add_imu("IMU", self.lander.scStateOutMsg)
        self.SM.add_star_tracker("ST", self.lander.scStateOutMsg)

        self.SM.register_to_task(self.sim, "record")  # Register task to the simulation environment

        sc_recorder = self.lander.scStateOutMsg.recorder()
        self.sim.AddModelToTask("record", sc_recorder)

        self.sc_recorder = sc_recorder

    def attach_scene(self,scene):
        self.scene = scene
        # State recorder of body
        self.sc_recorder = scene.getBody(self.lander.ModelTag).getOrigin().stateOutMsg.recorder()
        self.sim.AddModelToTask("record", self.sc_recorder)

    def output(self):
        """
        Minimal Basilisk bootstrap, this will need to be majorly rewritten when the simulation environment is more defined,
        IE when we actually have a Spacecraft.py and an Environment.py.
        :param num_steps: Number of timesteps to run
        """



        # Return all logged data
        imu_output = {}
        for name, imu in self.imus.items():
            imu_output[name] = {
                field: getattr(imu.recorder, field)
                for field in imu.fields
            }

        st_output = {}
        for name, st in self.star_trackers.items():
            st_output[name] = {
                field: getattr(st.recorder, field)
                for field in st.fields
            }





        return {
            "imu": imu_output,
            "star_tracker": st_output,
            "true_data": self.lander_true_status()
        }

    def lander_true_status(self):
        lander_pos = self.sc_recorder.r_CN_N
        lander_vel = self.sc_recorder.v_CN_N
        lander_attitude = self.sc_recorder.sigma_BN
        lander_angular_rate = self.sc_recorder.omega_BN_B
        #lander_mass = self.sc_recorder.mHub
        #lander_CoM = self.sc_recorder.r_BcB_B
        return [lander_pos, lander_vel, lander_attitude, lander_angular_rate]
