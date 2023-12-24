import matplotlib.pyplot as plt
import math
import numpy as np

class simulator:
    def __init__(self,xcoor_init,ycoor_init,xspeed_init,yspeed_init,G,M):
        self.xcoor = xcoor_init
        self.ycoor = ycoor_init
        self.xspeed = xspeed_init
        self.yspeed = yspeed_init
        self.G = G
        self.M = M
    def acc_vector(self,gravity):
        acc_angle = math.atan2(-self.ycoor,-self.xcoor)
        return gravity * math.cos(acc_angle), gravity * math.sin(acc_angle)
    def simulation(self,max_step,radius):
        step = 0
        xtracker = []
        ytracker = []
        altitude_tracker = []
        while step < max_step:
            length = self.xcoor**2 + self.ycoor**2
            gravity = self.G * self.M / length
            xacc,yacc = self.acc_vector(gravity)
            self.xspeed += xacc
            self.yspeed += yacc
            self.xcoor += self.xspeed
            self.ycoor += self.yspeed
            step += 1
            xtracker.append(self.xcoor)
            ytracker.append(self.ycoor)
            altitude = np.sqrt(self.xcoor**2 + self.ycoor**2) - radius
            altitude_tracker.append(altitude)
        return xtracker,ytracker,altitude_tracker

G = 6.674080e-11
M = 7.34767309e22
R = 1.7374e6
altitude = 1e5
init_speed = math.sqrt(G * M / (R + altitude))

xcoor,ycoor = [R + altitude,0]
xspeed,yspeed = [0,init_speed]

sim = simulator(xcoor,ycoor,xspeed,yspeed,G,M)
xtrack,ytrack,altitude_tracker = sim.simulation(5000, R)

angles = np.linspace(0,2 * np.pi,1000)
sine = R * np.sin(angles)
cosine = R * np.cos(angles)

plt.figure(1)
plt.plot(xtrack,ytrack)
plt.plot(cosine,sine)

plt.figure(2)
plt.plot(altitude_tracker)
plt.show()