from Basilisk.utilities import SimulationBaseClass, macros, vizSupport
from Basilisk.simulation import spacecraft
from Sensors.Sensors import SensorsManager

class Vehicle:
    def __init__(self, sampling_sec=0.01):
        self.imus = {}              # sensor IMU initialization, if you have multiple IMUs they get stored here
        self.star_trackers = {}     # sensor Star Tracker initialization, if you have multiple IMUs they get stored here
        self.lander = spacecraft.Spacecraft()        # Create a spacecraft instance from the spacecraft basilisk module
        self.lander.ModelTag = "SERVAL"              # Tag the spacecraft with a name

        self.lander.hub.r_CN_NInit = [70000, 0.0, 0.0]  # Current position within the inertial frame (in m)
        self.lander.hub.v_CN_NInit = [23500.0, 5000.0, 1000.0]  # Current velocity within the inertial frame (m/s)
        self.lander.hub.sigma_BNInit = [[0.0], [0.7], [0.0]]  # Current attitude with respect to the body and inertial frame utilizing a Modified Rodrigues Parameter (MRP) vector. (N->P)
        self.lander.hub.omega_BN_BInit = [[0.0], [0.0], [1.0]]  # Current angular velocity with body frame relative to inertial frame (rad/s)
        self.lander.hub.mHub = 2120.0                     #mass in kg
        self.lander.hub.r_BcB_B = [0,0,0]                   #Center of mass in B frame

        self.sampling_sec = sampling_sec
        self.sampling_ns = macros.sec2nano(self.sampling_sec)  # how often to sample sensors in ns

        self.loggers = {}                                   #Holds the logger instances

        self.SensorManager = SensorsManager(self, self.sampling_sec) #cause it expects it in seconds! Had that wrong before, whoopsie

        self.sc_recorder = None

    def run(self, num_steps: int = 1):
        """
        Minimal Basilisk bootstrap, this will need to be majorly rewritten when the simulation environment is more defined,
        IE when we actually have a Spacecraft.py and an Environment.py.
        :param num_steps: Number of timesteps to run
        """

        # Create simulation
        sim = SimulationBaseClass.SimBaseClass()                        # Initialize/instantiate a simulation environment
        process = sim.CreateNewProcess("proc")                          # Create a new simulation process
        task = sim.CreateNewTask("record", self.sampling_ns)      # Create a new task in the simulation
        process.addTask(task)                                               # Add created task to the process


        sim.AddModelToTask("record", self.lander)  # Add spacecraft to task

        # Add sensors to spacecraft
        self.SensorManager.add_imu("IMU", self.lander.scStateOutMsg)
        self.SensorManager.add_star_tracker("ST", self.lander.scStateOutMsg)

        self.SensorManager.register_to_task(sim, "record")  # Register task to the simulation environment

        sc_recorder = self.lander.scStateOutMsg.recorder()
        sim.AddModelToTask("record", sc_recorder)


        # Run one timestep
        sim.InitializeSimulation()  # Start the simulation
        sim.ConfigureStopTime(num_steps * self.sampling_ns)  # When the simulation should stop
        sim.ExecuteSimulation()

        self.sc_recorder = sc_recorder




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
