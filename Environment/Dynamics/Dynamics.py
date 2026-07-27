from Basilisk.architecture import sysModel, messaging
from Basilisk.utilities import RigidBodyKinematics as rbk
import numpy as np
from typing import Any, Sequence

from Basilisk.simulation import dynamicEffector

class ConstantForce(dynamicEffector.DynamicEffector):

    def __init__(self, force_N):
        super().__init__()

        self.force_N = np.array(force_N)

    def computeForceTorque(self, integTime, timeStep):

        # Current spacecraft attitude
        dcm_BN = rbk.MRP2C(self.hubSigma)

        # Convert to body frame
        self.forceExternal_B = dcm_BN @ self.force_N

        self.torqueExternalPntB_B[:] = [0.0,0.0,0.0]