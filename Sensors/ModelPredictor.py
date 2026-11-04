import numpy as np

class ModelPredictor:

    def __init__(self, inertia, initial_omega):
        # The inertia matrix and initial angular velocity of the rigid body 
        self.I = np.asarray(inertia, dtype=float).reshape((3, 3))
        self.I_inv = np.linalg.inv(self.I)
        self.omega = np.asarray(initial_omega, dtype=float).reshape((3, 1))

    def predict(self, torque, dt):
        torque = np.asarray(torque, dtype=float).reshape((3, 1))

        #Calculate Angualar Momentum: L = I * omega
        L = self.I @ self.omega

        #Cross product: omega x (I * omega)
        omega_cross_L = np.cross(self.omega.flatten(), L.flatten()).reshape((3,1))

        # Angular acceleration: omega_dot = I_inv * (torque - omega x (I * omega))
        omega_dot = self.I_inv @ (torque - omega_cross_L)

        # Update angular velocity: omega = omega + omega_dot * dt
        self.omega += omega_dot * dt

        return self.omega.flatten().copy()  # Return a copy to avoid external modifications
     
    def calculate_thruster_torque(self, engines, enabled_engines):

        total_torque = np.zeros(3)

        for name, engine in engines.items():

            # Ignore engines that are turned off
            if name not in enabled_engines:
                continue

            config = engine.config

            # Read thrust magnitude
            thrust = float(config.MaxThrust)

            # Read thrust direction
            direction = np.asarray(
                config.thrDir_B, dtype=float
            ).reshape(3)

            # Read thruster location
            location = np.asarray(
                config.thrLoc_B, dtype=float
            ).reshape(3)

            # Calculate force
            force = thrust * direction

            # Calculate torque
            torque = np.cross(location, force)

            # Add torque from this engine
            total_torque += torque

        return total_torque
