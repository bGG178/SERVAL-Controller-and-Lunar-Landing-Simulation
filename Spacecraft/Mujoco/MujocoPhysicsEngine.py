import mujoco
import numpy as np
from Sensors.Laser.Altimeter import LaserAltimeter
from Spacecraft.Mujoco.FrameTransforms import (
    MOON_RADIUS,
    TERRAIN_POSITION_MUJOCO,
    TERRAIN_ROTATION_MUJOCO,
    TERRAIN_SCALE,
)


OBJ_PATH = "Environment/Objects/SouthPole.obj"
def clean_obj(obj_path):
    with open(obj_path, "r", errors="replace") as f:
        lines = f.readlines()

    # Find valid vertices
    valid_vertex_indices = set()
    vertex_count = 0

    for line in lines:
        if line.startswith("v "):
            vertex_count += 1
            parts = line.split()

            try:
                x = float(parts[1])
                y = float(parts[2])
                z = float(parts[3])

                if np.isfinite(x) and np.isfinite(y) and np.isfinite(z):
                    valid_vertex_indices.add(vertex_count)

            except (ValueError, IndexError):
                pass

    # Build new OBJ
    output = []
    vertex_map = {}

    new_index = 1
    old_index = 0

    # Vertices
    for line in lines:
        if line.startswith("v "):
            old_index += 1

            if old_index in valid_vertex_indices:
                output.append(line)
                vertex_map[old_index] = new_index
                new_index += 1

        elif line.startswith("f "):
            parts = line.split()[1:]

            face_vertices = []

            try:
                for part in parts:
                    vertex_index = int(part.split("/")[0])

                    if vertex_index < 0:
                        vertex_index = vertex_count + vertex_index + 1

                    # If ANY vertex in the face is invalid,
                    # discard the entire face.
                    if vertex_index not in valid_vertex_indices:
                        face_vertices = []
                        break

                    face_vertices.append(vertex_map[vertex_index])

                if face_vertices:
                    output.append(
                        "f " + " ".join(map(str, face_vertices)) + "\n"
                    )

            except (ValueError, IndexError):
                pass

    return "".join(output).encode("utf-8")


def _xml_vec(values):
    return " ".join(str(float(value)) for value in values)


def initialize_mujoco(LF:int):

    obj_data = clean_obj(
        "Environment/Objects/SouthPole.obj"
    )

    XML= get_xml(LF)

    model = mujoco.MjModel.from_xml_string(
        XML,
        assets={
            "SouthPole.obj": obj_data
        }
    )

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


def get_xml(LF):
    if LF == 0:  # Creates a sphere for the altimeter instead of lunar terrain obj
        XML = f"""
            <mujoco model="laser_altimeter">

                <option gravity="0 0 -9.81"/>
                <statistic extent="{MOON_RADIUS * 2}"/>

                <worldbody>

                    <geom name="moon"
                          type="sphere"
                          size="{MOON_RADIUS}"
                          pos="0 0 0"/>

                    <body name="spacecraft" mocap="true" pos="0 0 {MOON_RADIUS + 1500}">

                        <geom name="spacecraft_body"
                              type="box"
                              size="0.5 0.5 0.2"/>

                        <site name="laser_altimeter"
                              pos="0 0 -0.2"
                              euler="90 0 0"
                              size="0.03"/>

                    </body>

                </worldbody>

            </mujoco>
            """
    elif LF == 1:  # Use lunar terrain OBJ
        XML = f"""
         <mujoco model="laser_altimeter">

            <compiler
                meshdir="."
                usethread="true"
            />

            <size
                memory="1G"
            />

            <asset>
                <mesh name="lunarTerrain"
                      file="SouthPole.obj"
                      scale="{_xml_vec(TERRAIN_SCALE)}"
                      maxhullvert="1000"/>
            </asset>

            <option gravity="0 0 -9.81"/>

            <statistic extent="50000"/>

            <worldbody>

                <geom name="lunarTerrain_geom"
                      type="mesh"
                      mesh="lunarTerrain"
                      pos="{_xml_vec(TERRAIN_POSITION_MUJOCO)}"
                      euler="{_xml_vec(TERRAIN_ROTATION_MUJOCO)}"
                      contype="1"
                      conaffinity="1"/>

                <body name="spacecraft" mocap="true">

                    <geom name="spacecraft_body"
                          type="box"
                          size="0.5 0.5 0.2"
                          contype="1"
                          conaffinity="1"/>

                    <site name="laser_altimeter"
                          pos="0 0 -0.2"
                          euler="90 0 0"
                          size="0.03"/>

                </body>

            </worldbody>

        </mujoco>
        """
    else:  # Use both a smooth spherical surface and lunar terrain obj
        XML = f"""
        <mujoco model="laser_altimeter">

            <asset>
                <mesh name="south_pole"
                      file="SouthPole.obj"
                      scale="{_xml_vec(TERRAIN_SCALE)}"/>
            </asset>

            <option gravity="0 0 0"/>
            <statistic extent="{MOON_RADIUS * 2}"/>

            <worldbody>

                <!-- Smooth spherical Moon reference -->
                <geom name="moon_sphere"
                      type="sphere"
                      size="{MOON_RADIUS}"
                      pos="0 0 0"

                      contype="1"
                        conaffinity="1"/>

                <!-- Actual lunar terrain -->
                <geom name="south_pole"
                      type="mesh"
                      mesh="south_pole"
                      pos="{_xml_vec(TERRAIN_POSITION_MUJOCO)}"
                      euler="{_xml_vec(TERRAIN_ROTATION_MUJOCO)}"
                    contype="1"
                     conaffinity="1"/>

                <!-- Spacecraft -->
                <body name="spacecraft" mocap="true">

                    <geom name="spacecraft_body"
                          type="box"
                          size="0.5 0.5 0.2"
                          contype="1"
                          conaffinity="1"/>

                    <site name="laser_altimeter"
                          pos="0 0 -0.2"
                          euler="90 0 0"
                          size="0.03"/>

                </body>

            </worldbody>

        </mujoco>
        """
    return XML
