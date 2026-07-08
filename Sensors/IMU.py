# IMU.py
from Basilisk.simulation import imuSensor

class IMU:
    def __init__(self, name="IMU"):
        self.name = name
        self.model = imuSensor.ImuSensor()

        #print(imuSensor.ImuSensor.__module__)               #Debug prints
        #print(dir(imuSensor.ImuSensor))                     #Debug prints

        # Default configuration
        self.model.ModelTag = name                          #Set model name
        self.model.sensorPos_B = [0.0, 0.0, 0.0]            #Position of the sensor on the vehicle in meters from COM
        self.model.dcm_PB = [                               #Vehicle DCM (What is a DCM?)
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0]
        ]
        self.model.senRotBias   = [0.0001, 0.0001, 0.0001]  #Constant offset added to true angular rate of IMU measurement, pre-noise
        self.model.senTransBias = [0.001,  0.001,  0.001]   #Accelerometer offset added to true acceleration of IMU before noise
        self.model.accelScale   = [1.0, 1.0, 1.0]           # Gain error multiplied across true acceleration
        self.model.gyroScale    = [1.0, 1.0, 1.0]           # Gain error multiplied across true angular rate

        self.model.navErrorsAccel = [0.0005, 0.0005, 0.0005]  # m/s^2, instantaneous gaussian noise
        self.model.navErrorsGyro = [0.00001, 0.00001, 0.00001]  # rad/s, white noise standard deviation

        self.model.setWalkBoundsAccel([1e-5, 1e-5, 1e-5])  # m/s^2 per sqrt(sec), bias drift over time.
        self.model.setWalkBoundsGyro([1e-6, 1e-6, 1e-6])  # rad/s per sqrt(sec), bias drift over time.

        self.model.setErrorBoundsAccel([0.01, 0.01, 0.01]) #maximum drift magnitude
        self.model.setErrorBoundsGyro([0.001, 0.001, 0.001]) #maximum drift magnitude

        self.model.PMatrixAccel = [ #acc process noise covariance, how quickly bias drift accumulates
            [.0010, 0.0, 0.0],
            [0.0, .0010, 0.0],
            [0.0, 0.0, .0010]
        ]

        self.model.PMatrixGyro = [ #gyro process noise covariance, how quickly bias drift accumulates
            [.0010, 0.0, 0.0],
            [0.0, .0010, 0.0],
            [0.0, 0.0, .0010]
        ]

        self.fields = [
            "DVFramePlatform",
            "AccelPlatform",
            "DRFramePlatform",
            "AngVelPlatform"
        ]

        self.recorder = None                                #Initialize the logger/recorder variable

    # ------------------------------------------------------------------
    # Called by sensors.py
    # ------------------------------------------------------------------

    def connect_state(self, sc_state_msg):
        """
        Subscribes the IMU to the state (?)
        :param sc_state_msg: Variable containing the state message subscription.
        :return:
        """
        self.model.scStateInMsg.subscribeTo(sc_state_msg)

    def attach_recorder(self, sampling_ns):
        """
        Attaches the recorder to the IMU
        :param sampling_ns: Sampling frequency of the IMU
        :return:
        """

        self.recorder = self.model.sensorOutMsg.recorder(sampling_ns)
        return self.recorder

    def configure(self, cfg):
        """
        config dict from sensors.py to overwrite this default one
        :param cfg: Optional config dict from sensors.py
        """
        for key, val in cfg.items():
            setattr(self.model, key, val)#Set attr used to modify the config

    def process(self, env):
        """
        Black-box processing step
        :param env: environment variable
        """
        pass  # placeholder for future non-Basilisk processing

    def output(self, index):
        """Return IMU output at a given timestep (index)"""
        return {
            "DVFramePlatform": self.recorder.DVFramePlatform[index],
            "AccelPlatform":   self.recorder.AccelPlatform[index],
            "DRFramePlatform": self.recorder.DRFramePlatform[index],
            "AngVelPlatform":  self.recorder.AngVelPlatform[index],
        }
