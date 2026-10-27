
# FaultInjection.py
# SERVAL Sensor Fault Injection Module
#
# Supported faults:
# 1. Transient spike
# 2. Step fault
# 3. Permanent stuck fault
# 4. Gradual drift
#
# Supports:
# - IMU angular velocity
# - IMU acceleration
# - Star tracker attitude (MRP representation)
#
# Time is measured in seconds.
# Axis: 0 = X, 1 = Y, 2 = Z

import numpy as np


class FaultInjector:

    def __init__(self):
        self.faults = []
        self.stuck_measurements = {}

    # --------------------------------------------------
    # 1. TRANSIENT SPIKE
    # --------------------------------------------------

    def transient_spike(self, measurement, time,
                        start_time, duration, magnitude):

        if start_time <= time < start_time + duration:
            faulty_measurement = measurement + magnitude
        else:
            faulty_measurement = measurement

        return faulty_measurement

    # --------------------------------------------------
    # 2. STEP FAULT
    # --------------------------------------------------

    def step_fault(self, measurement, time,
                   start_time, magnitude):

        if time >= start_time:
            faulty_measurement = measurement + magnitude
        else:
            faulty_measurement = measurement

        return faulty_measurement

    # --------------------------------------------------
    # 3. PERMANENT STUCK FAULT
    # --------------------------------------------------

    def permanent_fault(self, measurement, time,
                        start_time, sensor_key):

        if time < start_time:
            self.stuck_measurements[sensor_key] = float(
                measurement
            )
            return measurement

        if sensor_key not in self.stuck_measurements:
            self.stuck_measurements[sensor_key] = float(
                measurement
            )

        return self.stuck_measurements[sensor_key]

    # --------------------------------------------------
    # 4. GRADUAL DRIFT
    # --------------------------------------------------

    def gradual_fault(self, measurement, time,
                      start_time, drift_rate):

        if time >= start_time:
            elapsed_time = time - start_time
            error = drift_rate * elapsed_time
            faulty_measurement = measurement + error
        else:
            faulty_measurement = measurement

        return faulty_measurement

    # --------------------------------------------------
    # 5. CONFIGURE FAULTS
    # --------------------------------------------------

    def add_fault(self, sensor, fault_type,
                  start_time, magnitude=0.0,
                  duration=None, axis=0,
                  field="AngVelPlatform"):

        valid_types = (
            "spike", "step", "permanent", "gradual"
        )

        if fault_type not in valid_types:
            raise ValueError(
                f"Invalid fault type: {fault_type}"
            )

        if axis not in (0, 1, 2):
            raise ValueError(
                "Axis must be 0, 1, or 2"
            )

        if fault_type == "spike":
            if duration is None or duration <= 0:
                raise ValueError(
                    "Spike requires a positive duration"
                )

        fault = {
            "sensor": sensor,
            "type": fault_type,
            "start_time": start_time,
            "magnitude": magnitude,
            "duration": duration,
            "axis": axis,
            "field": field
        }

        self.faults.append(fault)

    # --------------------------------------------------
    # 6. APPLY FAULTS TO SENSOR DATA
    # --------------------------------------------------

    def inject(self, sensor_data, time, sensor_name):

        # Copy data without modifying original output
        faulty_data = {
            key: np.array(value, copy=True)
            for key, value in sensor_data.items()
        }

        for i, fault in enumerate(self.faults):

            if fault["sensor"] != sensor_name:
                continue

            field = fault["field"]
            axis = fault["axis"]
            start = fault["start_time"]
            magnitude = fault["magnitude"]
            duration = fault["duration"]

            if field not in faulty_data:
                raise ValueError(
                    f"Field '{field}' not found "
                    f"for sensor '{sensor_name}'"
                )

            if faulty_data[field].shape != (3,):
                raise ValueError(
                    f"Field '{field}' must be a "
                    "three-component measurement"
                )

            measurement = float(
                faulty_data[field][axis]
            )

            # Transient spike
            if fault["type"] == "spike":

                measurement = self.transient_spike(
                    measurement,
                    time,
                    start,
                    duration,
                    magnitude
                )

            # Step fault
            elif fault["type"] == "step":

                measurement = self.step_fault(
                    measurement,
                    time,
                    start,
                    magnitude
                )

            # Permanent stuck fault
            elif fault["type"] == "permanent":

                sensor_key = (
                    sensor_name, field, axis, i
                )

                measurement = self.permanent_fault(
                    measurement,
                    time,
                    start,
                    sensor_key
                )

            # Gradual drift
            elif fault["type"] == "gradual":

                measurement = self.gradual_fault(
                    measurement,
                    time,
                    start,
                    magnitude
                )

            faulty_data[field][axis] = measurement

        return faulty_data

    # --------------------------------------------------
    # 7. IMU FAULT INJECTION
    # --------------------------------------------------

    def inject_imu(self, imu_data, time,
                   sensor_name="IMU"):

        return self.inject(
            imu_data,
            time,
            sensor_name
        )

    # --------------------------------------------------
    # 8. STAR TRACKER FAULT INJECTION
    # --------------------------------------------------

    def inject_star_tracker(self, star_data, time,
                            sensor_name="StarTracker"):

        return self.inject(
            star_data,
            time,
            sensor_name
        )

    # --------------------------------------------------
    # 9. CLEAR FAULTS
    # --------------------------------------------------

    def clear_faults(self):

        self.faults = []
        self.stuck_measurements = {}


# ======================================================
# EXAMPLE FAULT CONFIGURATION
# ======================================================

def create_default_faults():

    injector = FaultInjector()

    # IMU transient spike
    # X-axis gyro, 10.0 to 10.1 seconds
    injector.add_fault(
        sensor="IMU",
        fault_type="spike",
        start_time=10.0,
        duration=0.1,
        magnitude=0.5,
        axis=0,
        field="AngVelPlatform"
    )

    # IMU step fault
    # Y-axis gyro, beginning at 20 seconds
    injector.add_fault(
        sensor="IMU",
        fault_type="step",
        start_time=20.0,
        magnitude=0.2,
        axis=1,
        field="AngVelPlatform"
    )

    # IMU permanent stuck fault
    # Z-axis gyro, beginning at 30 seconds
    injector.add_fault(
        sensor="IMU",
        fault_type="permanent",
        start_time=30.0,
        axis=2,
        field="AngVelPlatform"
    )

    # IMU gradual drift
    # X-axis gyro, beginning at 40 seconds
    injector.add_fault(
        sensor="IMU",
        fault_type="gradual",
        start_time=40.0,
        magnitude=0.001,
        axis=0,
        field="AngVelPlatform"
    )

    # Star tracker transient attitude spike
    # Example for a three-component MRP field
    injector.add_fault(
        sensor="StarTracker",
        fault_type="spike",
        start_time=15.0,
        duration=0.1,
        magnitude=0.01,
        axis=0,
        field="sigma_BN"
    )

    # Star tracker gradual attitude drift
    injector.add_fault(
        sensor="StarTracker",
        fault_type="gradual",
        start_time=50.0,
        magnitude=0.0001,
        axis=1,
        field="sigma_BN"
    )

    return injector


