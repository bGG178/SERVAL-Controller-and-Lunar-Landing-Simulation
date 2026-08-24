import mujoco
import numpy as np
from scipy.spatial.transform import Rotation
from Sensors.Laser.Altimeter import LaserAltimeter


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


MOON_RADIUS = 1737400.0

TERRAIN_ROTATION_MUJOCO = [90.0, 90.0, 0.0]

# Convert MuJoCo XYZ Euler rotation to Vizard 3-2-1 Euler angles
TERRAIN_ROTATION_VIZARD = (
    Rotation.from_euler(
        "xyz",
        TERRAIN_ROTATION_MUJOCO,
        degrees=True
    )
    .as_euler(
        "zyx",
        degrees=False
    )[::-1]
    .tolist()
)
TERRAIN_SCALE = [10.0, 10.0, 10.0]


def initialize_mujoco(LF:int):

    obj_data = clean_obj(
        "Environment/Objects/SouthPole.obj"
    )

    if LF==0: #Creates a sphere for the altimeter instead of lunar terrain obj
        XML = f"""
            <mujoco model="laser_altimeter">

                <option gravity="0 0 -9.81"/>
                <statistic extent="{MOON_RADIUS * 2}"/>

                <worldbody>

                    <geom name="moon"
                          type="sphere"
                          size="{MOON_RADIUS}"
                          pos="0 0 0"/>

                    <body name="spacecraft" pos="0 0 {MOON_RADIUS + 1500}">

                        <freejoint/>

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
    elif LF==1: #Use lunar terrain OBJ
        XML = f"""
        <mujoco model="laser_altimeter">
    
            <asset>
                <mesh name="south_pole"
                      file="SouthPole.obj"
                      scale="{TERRAIN_SCALE[0]}
                             {TERRAIN_SCALE[1]}
                             {TERRAIN_SCALE[2]}"/>
            </asset>
    
            <option gravity="0 0 -9.81"/>
            <statistic extent="500000"/>
    
            <worldbody>
    
                <geom name="south_pole"
                      type="mesh"
                      mesh="south_pole"
                      pos="0 0 {-MOON_RADIUS}"
                      euler="{TERRAIN_ROTATION_MUJOCO[0]}
                             {TERRAIN_ROTATION_MUJOCO[1]}
                             {TERRAIN_ROTATION_MUJOCO[2]}"
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
    else: #Use both a smooth spherical surface and lunar terrain obj
        XML = f"""
        <mujoco model="laser_altimeter">

            <asset>
                <mesh name="south_pole"
                      file="SouthPole.obj"
                      scale="{TERRAIN_SCALE[0]}
                             {TERRAIN_SCALE[1]}
                             {TERRAIN_SCALE[2]}"/>
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
                      pos="0 0 {-MOON_RADIUS}"
                      euler="{TERRAIN_ROTATION_MUJOCO[0]}
                             {TERRAIN_ROTATION_MUJOCO[1]}
                             {TERRAIN_ROTATION_MUJOCO[2]}"
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

    spacecraft_geom_id = mujoco.mj_name2id(
        model,
        mujoco.mjtObj.mjOBJ_GEOM,
        "spacecraft_body"
    )
    print(spacecraft_geom_id)

    return altimeter