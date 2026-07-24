#This file will host the Ephemeride data from NASA SPICE as well as the class for planets/gravitational bodies
from Basilisk.simulation import spacecraft
from Basilisk.utilities import simIncludeGravBody



class CelestialBody():
    def __init__(self):
        self.gravFactory = simIncludeGravBody.gravBodyFactory()
        self.body = None
        self.spice = None
    def setBody(self, body):
        if body is "Earth":
            self.gravFactory.createEarth()
        if body is "Moon":
            self.gravFactory.createMoon()
        if body is "Sun":
            self.gravFactory.createSun()
    def loadSpice(self, sim,start_time):
        self.spice = self.gravFactory.createSpiceInterface(path="SPICE", time=start_time)
        sim.AddModelToTask("record", self.spice)
    def attachSpacecraft(self, spacecraft):
        self.gravFactory.addBodiesTo(spacecraft)

