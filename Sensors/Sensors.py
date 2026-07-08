from Basilisk.utilities import SimulationBaseClass, macros
from Basilisk.simulation import spacecraft

from IMU import IMU
from Sensors import StarTracker


class SensorsManager:
    def __init__(self, sampling_sec=0.5):             #2 Hz
        self.sampling_sec = sampling_sec                    #how often to sample sensors in s
        self.sampling_ns = macros.sec2nano(sampling_sec)    #how often to sample sensors in ns

        self.imus = {}                                      #sensor IMU initialization, if you have multiple IMUs they get stored here
        self.star_trackers = {}                             #sensor Star Tracker initialization, if you have multiple IMUs they get stored here
        self.loggers = {}                                   #Holds the logger instances

    def add_imu(self, name, sc_state_msg, config=None):
        """
        Mainly just makes an instance of the IMU in the simulation
        :param name: ID Of the IMU
        :param sc_state_msg: Basilisk State Message (? What is a state message)
        :param config: Any additional configuration that is required (?)
        :return:
        """
        imu = IMU(name)                                     #creates an instance of the IMU class
        imu.connect_state(sc_state_msg)                     #connects IMU input message to the spacecraft state output message (??)
        if config:
            imu.configure(config)                           #add additional config parameters such as noise, bias, and FOV (for some sensors).

        rec = imu.attach_recorder(self.sampling_ns)         #Initialize the recorder/logger for the imu
        self.imus[name] = imu                               #store IMU in the array of IMUs on board
        self.loggers[f"imu:{name}"] = rec                   #store logger in the array of loggers for the craft
        return imu                                          #returns the IMU instance

    def add_star_tracker(self, name, sc_state_msg, config=None):
        """
        Mainly just makes an instance of the Star Tracker in the simulation
        :param name: ID Of the ST
        :param sc_state_msg: Basilisk State Message (? What is a state message)
        :param config: Any additional configuration that is required (?)
        :return:
        """
        st = StarTracker(name)                              #creates an instance of the StarTracker class
        st.connect_state(sc_state_msg)                      #connects IMU input message to the spacecraft state output message (??)
        if config:
            st.configure(config)                            #add additional config parameters such as noise, bias, and FOV (for some sensors).

        rec = st.attach_recorder(self.sampling_ns)          #Initialize the recorder/logger for the ST
        self.star_trackers[name] = st                       #store ST in the array of STs on board
        self.loggers[f"star:{name}"] = rec                  #store logger in the array of loggers for the craft
        return st                                           #returns the ST instance

    def register_to_task(self, sim, task_name):
        """
        Registers the basilisk modules to the sensors
        :param sim: Simulation environment variable
        :param task_name: Name/ID of the task
        :return:
        """
        for imu in self.imus.values():
            sim.AddModelToTask(task_name, imu.model)    #Add the model of the sensor to the simulation task variable
            sim.AddModelToTask(task_name, imu.recorder) #Add recorder to the simulation task variable

        for st in self.star_trackers.values():
            sim.AddModelToTask(task_name, st.model)     #Add the model of the sensor to the simulation task variable
            sim.AddModelToTask(task_name, st.recorder)  #Add recorder to the simulation task variable

    def output(self, index):
        """
        Formats and returns the sensor output
        :param index: The index to categorize sensor logs
        :return:
        """
        return {
            "imu": {name: imu.output(index) for name, imu in self.imus.items()},                #IMU Logging and Output
            "star_tracker": {name: st.output(index) for name, st in self.star_trackers.items()} #ST Logging and Output
        }

    def run_single_timestep_test(self, num_steps:int=1):
        """
        Minimal Basilisk bootstrap, this will need to be majorly rewritten when the simulation environment is more defined,
        IE when we actually have a Spacecraft.py and an Environment.py.
        :param num_steps: Number of timesteps to run
        """

        #Create simulation
        sim = SimulationBaseClass.SimBaseClass()                    #Initialize/instantiate a simulation environment
        process = sim.CreateNewProcess("proc")                      #Create a new simulation process
        task = sim.CreateNewTask("task", self.sampling_ns) #Create a new task in the simulation
        process.addTask(task)                                       #Add created task to the process

        #Create static spacecraft (This will move into the Spacecraft.py when we get to making that file.
        lander = spacecraft.Spacecraft()                                #Create a spacecraft instance from the spacecraft basilisk module
        lander.ModelTag = "TestSC"                                      #Tag the spacecraft with a name

        lander.hub.r_CN_NInit = [7000e3, 0.0, 0.0]                      #Current position within the inertial frame (in km)
        lander.hub.v_CN_NInit = [0.0, 7.5e3, 0.0]                       #Current velocity within the inertial frame (km/s)
        lander.hub.sigma_BNInit = [[0.0], [0.0], [0.0]]                 #Current attitude with respect to the body and inertial frame utilizing a Modified Rodrigues Parameter (MRP) vector. (N->P)
        lander.hub.omega_BN_BInit = [[0.0], [0.0], [0.0]]               #Current angular velocity with body frame relative to inertial frame (rad/s)

        sim.AddModelToTask("task", lander)                     #Add spacecraft to task

        #Add sensors to spacecraft
        self.add_imu("IMU", lander.scStateOutMsg)
        self.add_star_tracker("ST", lander.scStateOutMsg)

        self.register_to_task(sim, "task")                 #Register task to the simulation environment

        #Run one timestep
        sim.InitializeSimulation()                                  #Start the simulation
        sim.ConfigureStopTime(num_steps * self.sampling_ns)         #When the simulation should stop
        sim.ExecuteSimulation()                                     #Run simulation

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
            "star_tracker": st_output
        }
