import os
from Basilisk.utilities import vizSupport
import Spacecraft.Mujoco.MujocoPhysicsEngine as MPE
def initialize_vizard(sim, sc):

    if vizSupport.vizFound:

        # Create Vizard with both the real spacecraft and terrain
        viz = vizSupport.enableUnityVisualization(
            sim,
            "record",
            [sc.lander, sc.terrain],
            saveFile=__file__,
        )

        viz.settings.showSpacecraftLabels = 1
        viz.settings.showSpacecraftAsSprites = -1
        viz.settings.ambient = 0.5
        viz.settings.spacecraftShadowBrightness = 0.07

        vizSupport.setActuatorGuiSetting(viz)

        # Real spacecraft model
        model_path = os.path.abspath(
            os.path.join("Spacecraft", "IM1.obj")
        )

        vizSupport.createCustomModel(
            viz,
            model_path,
            simBodiesToModify=[sc.lander.ModelTag],
            offset=[0.73, -0.73, -1.05],
            scale=[0.0004, 0.0004, 0.0004]
        )

        # Lunar terrain model
        terrain_path = os.path.abspath(
            os.path.join(
                "Environment",
                "Objects",
                "SouthPole.obj"
            )
        )

        vizSupport.createCustomModel(
            viz,
            terrain_path,
            simBodiesToModify=[sc.terrain.ModelTag],
            offset=[0.0, 0.0, 0.0],
            rotation=MPE.TERRAIN_ROTATION_VIZARD,
            scale=MPE.TERRAIN_SCALE
        )

        print([
            x for x in dir(viz.settings)
            if "planet" in x.lower()
               or "celestial" in x.lower()
               or "moon" in x.lower()
        ])

        viz.settings.forceStartAtSpacecraftLocalView = 1

        viz.settings.scViewToPlanetViewBoundaryMultiplier = 10
        viz.settings.planetViewToHelioViewBoundaryMultiplier = 10

