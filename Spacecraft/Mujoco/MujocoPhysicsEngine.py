import mujoco
import numpy as np
import time
from Sensors.Laser.Altimeter import LaserAltimeter

XML = """
<mujoco model="laser_altimeter">

    <option gravity="0 0 -9.81"/>
    <statistic extent="500000"/>

    <worldbody>

        <!-- Ground -->
        <geom name="ground"
              type="plane"
              size="100000 100000 0.1"
              pos="0 0 0"
              rgba="0.3 0.3 0.3 1"/>

        <!-- Spacecraft -->
        <body name="spacecraft" pos="0 0 1500">

            <freejoint/>

            <geom name="spacecraft_body"
                  type="box"
                  size="0.5 0.5 0.2"
                  rgba="0.1 0.4 0.8 1"/>

            <!--
                Laser altimeter.

                MuJoCo ray direction is the site's local +Z axis.
                We rotate the site 180 degrees about X so +Z points
                downward in the world frame.
            -->
            <site name="laser_altimeter"
                  pos="0 0 -0.2"
                  euler="180 0 0"
                  size="0.03"
                  rgba="1 0 0 1"/>

        </body>

    </worldbody>
</mujoco>
"""

altimeter = None
model = None
data = None
laser_id = None

def initialize_mujoco():
    # ============================================================
    # Load model
    # ============================================================

    model = mujoco.MjModel.from_xml_string(XML)
    data = mujoco.MjData(model)

    laser_id = mujoco.mj_name2id(
        model,
        mujoco.mjtObj.mjOBJ_SITE,
        "laser_altimeter"
    )

    altimeter = LaserAltimeter(
        model,
        data,
        laser_id
    )
    return altimeter


