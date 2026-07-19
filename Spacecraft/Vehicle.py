from Basilisk.utilities import SimulationBaseClass, macros
from Basilisk.simulation import spacecraft
from Sensors.Sensors import SensorsManager

class Vehicle:
    def __init__(self, sampling_sec=0.01):
        self.imus = {}              # sensor IMU initialization, if you have multiple IMUs they get stored here
        self.star_trackers = {}     # sensor Star Tracker initialization, if you have multiple IMUs they get stored here
        self.lander = spacecraft.Spacecraft()        # Create a spacecraft instance from the spacecraft basilisk module
        self.lander.ModelTag = "SERVAL"              # Tag the spacecraft with a name

        self.lander.hub.r_CN_NInit = [7000e3, 0.0, 0.0]  # Current position within the inertial frame (in m)
        self.lander.hub.v_CN_NInit = [0.0, 0.0, 0.0]  # Current velocity within the inertial frame (m/s)
        self.lander.hub.sigma_BNInit = [[0.0], [0.0], [0.0]]  # Current attitude with respect to the body and inertial frame utilizing a Modified Rodrigues Parameter (MRP) vector. (N->P)
        self.lander.hub.omega_BN_BInit = [[3.0], [0.0], [0.0]]  # Current angular velocity with body frame relative to inertial frame (rad/s)


        self.sampling_sec = sampling_sec
        self.sampling_ns = macros.sec2nano(self.sampling_sec)  # how often to sample sensors in ns

        self.loggers = {}                                   #Holds the logger instances

        self.SensorManager = SensorsManager(self, self.sampling_sec) #cause it expects it in seconds! Had that wrong before, whoopsie

    def run(self, num_steps: int = 1):
        """
        Minimal Basilisk bootstrap, this will need to be majorly rewritten when the simulation environment is more defined,
        IE when we actually have a Spacecraft.py and an Environment.py.
        :param num_steps: Number of timesteps to run
        """

        # Create simulation
        sim = SimulationBaseClass.SimBaseClass()                        # Initialize/instantiate a simulation environment
        process = sim.CreateNewProcess("proc")                          # Create a new simulation process
        task = sim.CreateNewTask("task", self.sampling_ns)      # Create a new task in the simulation
        process.addTask(task)                                               # Add created task to the process


        sim.AddModelToTask("task", self.lander)  # Add spacecraft to task

        # Add sensors to spacecraft
        self.SensorManager.add_imu("IMU", self.lander.scStateOutMsg)
        self.SensorManager.add_star_tracker("ST", self.lander.scStateOutMsg)

        self.SensorManager.register_to_task(sim, "task")  # Register task to the simulation environment

        sc_recorder = self.lander.scStateOutMsg.recorder()
        sim.AddModelToTask("task", sc_recorder)

        # Run one timestep
        sim.InitializeSimulation()  # Start the simulation
        sim.ConfigureStopTime(num_steps * self.sampling_ns)  # When the simulation should stop
        sim.ExecuteSimulation()


        print("Omega history:")
        print(sc_recorder.omega_BN_B)

        print("Attitude history:")
        print(sc_recorder.sigma_BN)

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

        print(len(st_output["ST"]["qInrtl2Case"]))
        print(len(imu_output["IMU"]["AccelPlatform"]))


        return {
            "imu": imu_output,
            "star_tracker": st_output
        }
