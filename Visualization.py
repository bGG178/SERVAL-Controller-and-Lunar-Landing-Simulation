import os
from Basilisk.utilities import vizSupport


def initialize_vizard(sim, sc):

    """
    Just start up vizard sim support, will output file to _VizFiles folder
    :param sim:
    :param sc:
    :return:
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




        #vizSupport.createCustomModel(viz,
        #                             # Specifying relative model path is useful for sharing scenarios and resources:
        #                             #modelPath=r"C:\Users\welov\Downloads\SERVALCONTROLLER_OVERARCH\Environment\Objects\ItokawaHayabusa.obj",#If you wanted to use a custom object you can here, then uncomment below
        #                             shader=1,
        #                             simBodiesToModify=['Moon'],
        #                             #scale=[962, 962, 962]) # This may have to be tweaked
        #                             )
        vizSupport.setActuatorGuiSetting(viz)

        model_path = os.path.abspath(
            os.path.join("Spacecraft", "IM1.obj")
        )

        vizSupport.createCustomModel(
            viz,
            model_path,
            offset=[0.73, -0.73, -1.05], #(last value is up and down but reverse so negative = up)
            scale=[0.0004,0.0004,0.0004] #Approximately correct scaling of the model
        )