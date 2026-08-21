import os
from Basilisk.utilities import vizSupport


def initialize_vizard(sim, sc):

    """
    Start Vizard visualization.
    """

    if vizSupport.vizFound:

        viz = vizSupport.enableUnityVisualization(
            sim,
            "record",
            sc.lander,
            saveFile=__file__,
        )

        viz.settings.showSpacecraftLabels = 1
        viz.settings.showSpacecraftAsSprites = -1
        viz.settings.ambient = 0.5
        viz.settings.spacecraftShadowBrightness = 0.07

        vizSupport.setActuatorGuiSetting(viz)

        # ---------------------------------------------------------
        # Spacecraft
        # ---------------------------------------------------------

        model_path = os.path.abspath(
            os.path.join("Spacecraft", "IM1.obj")
        )

        vizSupport.createCustomModel(
            viz,
            model_path,
            offset=[0.73, -0.73, -1.05],
            scale=[0.0004, 0.0004, 0.0004]
        )

        # ---------------------------------------------------------
        # South Pole terrain
        # ---------------------------------------------------------

        south_pole_path = os.path.abspath(
            os.path.join(
                "Environment",
                "Objects",
                "SouthPole.obj"
            )
        )

        vizSupport.createCustomModel(
            viz,
            south_pole_path,
            offset=[100, 0, 0],
            scale=[1, 1, 1]
        )