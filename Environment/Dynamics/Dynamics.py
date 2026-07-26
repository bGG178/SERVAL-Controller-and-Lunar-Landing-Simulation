from Basilisk.architecture import sysModel, messaging
from Basilisk.utilities import RigidBodyKinematics as rbk
import numpy as np
from typing import Any, Sequence


class ConstantGravity(sysModel.SysModel):
    """
    A class to model constant gravity force in a simulation.

    Attributes:
        force_N (Sequence[float]):
            The constant gravitational force vector in the inertial frame (N frame).
        frameInMsg (messaging.SCStatesMsgReader):
            Reader for spacecraft state messages.
        forceOutMsg (messaging.ForceAtSiteMsg):
            Message to output the computed force in the body-fixed frame (B frame).
    """

    def __init__(self, force_N: Sequence[float], *args: Any):
        """
        Args:
            force_N (Sequence[float]): The gravity force vector in the
            inertial reference frame.
        """
        super().__init__(*args)

        self.force_N = force_N

        self.frameInMsg = messaging.SCStatesMsgReader()

        self.forceOutMsg = messaging.ForceAtSiteMsg()

    def UpdateState(self, CurrentSimNanos: int):
        """Called at every integrator step to compute the force
        in the spacecraft-fixed reference frame."""
        # N frame: inertial frame
        # B frame: body-fixed frame
        frame: messaging.SCStatesMsgPayload = self.frameInMsg()
        dcm_BN = rbk.MRP2C(frame.sigma_BN)
        force_B = np.dot(dcm_BN, self.force_N)

        payload = messaging.ForceAtSiteMsgPayload(force_S=force_B)
        self.forceOutMsg.write(payload, CurrentSimNanos, self.moduleID)
