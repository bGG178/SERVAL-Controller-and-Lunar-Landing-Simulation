#This file will host the Ephemeride data from NASA SPICE as well as the class for planets/gravitational bodies
from Basilisk.simulation import spacecraft
from Basilisk.utilities import simIncludeGravBody



class CelestialBody():
    def __init__(self):
        self.gravFactory = simIncludeGravBody.gravBodyFactory()
        self.bodies = {}
        self.spice = None

    def createBody(self, name, central=False):
        body = self.gravFactory.createMoon()
        body.isCentralBody = central
        self.bodies[name] = body
        return body

    def attachTo(self, spacecraft):
        self.gravFactory.addBodiesTo(spacecraft)

    def loadSpice(self, sim, startTime, path="SPICE"):
        self.spice = self.gravFactory.createSpiceInterface(
            path=path,
            time=startTime
        )
        sim.AddModelToTask("record", self.spice)

