from Basilisk.architecture import sysModel, messaging
from Basilisk.utilities import RigidBodyKinematics as rbk
import numpy as np
from typing import Any, Sequence


class ThrusterVizMessageWriter(sysModel.SysModel):
    """Publish a ``THROutputMsg`` from a MuJoCo scalar thrust command."""

    def __init__(
        self,
        thrusterName: str,
        thrustInMsg: messaging.SingleActuatorMsg,
        maxThrust: float,
        thrusterLocation: Sequence[float],
        thrusterDirection: Sequence[float],
        visualizationScale: float,
        *args: Any,
    ):
        """Create a Vizard thruster message writer.

        :param thrusterName: Name of the MuJoCo actuator represented in Vizard.
        :param thrustInMsg: Scalar thrust command message used by MuJoCo.
        :param maxThrust: Nominal maximum thrust for Vizard scaling.
        :param thrusterLocation: Thruster location in the attached body frame.
        :param thrusterDirection: Unit thrust direction in the attached body frame.
        :param visualizationScale: Scale factor applied only to the Vizard thrust.
        """
        super().__init__(*args)
        self.ModelTag = thrusterName
        self.maxThrust = abs(maxThrust)  # [N]
        self.thrusterLocation = list(thrusterLocation)
        self.thrusterDirection = list(thrusterDirection)
        self.visualizationScale = visualizationScale
        self.thrustInMsg = messaging.SingleActuatorMsgReader()
        self.thrustInMsg.subscribeTo(thrustInMsg)
        self.thrOutMsg = messaging.THROutputMsg()

    def Reset(self, CurrentSimNanos: int):
        """Write the initial thruster visualization payload."""
        self._write_thruster_payload(CurrentSimNanos)

    def UpdateState(self, CurrentSimNanos: int):
        """Write the current thruster visualization payload."""
        self._write_thruster_payload(CurrentSimNanos)

    def _write_thruster_payload(self, CurrentSimNanos: int):
        """Write the current scalar thrust command for Vizard."""
        thrustForce = self.thrustInMsg().input  # [N]
        vizThrustForce = self.visualizationScale * thrustForce  # [N]
        payload = messaging.THROutputMsgPayload()
        payload.maxThrust = self.maxThrust
        payload.thrustForce = vizThrustForce
        if self.maxThrust > 0.0:
            payload.thrustFactor = vizThrustForce / self.maxThrust
        payload.thrustBlowDownFactor = 1.0
        payload.ispBlowDownFactor = 1.0
        payload.thrusterLocation = self.thrusterLocation
        payload.thrusterDirection = self.thrusterDirection
        payload.thrustForce_B = [
            vizThrustForce * directionComponent
            for directionComponent in self.thrusterDirection
        ]
        self.thrOutMsg.write(payload, CurrentSimNanos, self.moduleID)


