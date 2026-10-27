
# FaultInjection.py
# SERVAL Sensor Fault Injection Module

class FaultInjector:

    def __init__(self):
        self.faults = []

    def transient_spike(self, measurement, time,
                        start_time, duration, magnitude):

        # Check if the fault is active
        if start_time <= time < start_time + duration:
            faulty_measurement = measurement + magnitude

        else:
            faulty_measurement = measurement

        return faulty_measurement


    def step_fault(self, measurement, time,
                   start_time, magnitude):

        # Check if the fault has started
        if time >= start_time:
            faulty_measurement = measurement + magnitude

        else:
            faulty_measurement = measurement

        return faulty_measurement


    def permanent_fault(self, measurement, time,
                        start_time):

        # Before the fault, update the stored measurement
        if time < start_time:
            self.last_measurement = measurement
            return measurement

        # When the fault starts, freeze the sensor
        if not hasattr(self, "stuck_measurement"):
            self.stuck_measurement = getattr(
                self, "last_measurement", measurement
            )

        return self.stuck_measurement


    def gradual_fault(self, measurement, time,
                      start_time, drift_rate):

        # Check if the fault has started
        if time >= start_time:

            # Calculate time since the fault began
            elapsed_time = time - start_time

            # Calculate accumulated sensor error
            error = drift_rate * elapsed_time

            # Add the error to the measurement
            faulty_measurement = measurement + error

        else:
            faulty_measurement = measurement

        return faulty_measurement

