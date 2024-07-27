import numpy as np
import matplotlib.pyplot as plt
import json

g = 6.67e-11

def compute_orbit_velocity(center_mass,orbit):
    return np.sqrt(g * center_mass / orbit)

with open("presets.json","r") as f:
    property = json.load(f)

class object:
    #Basic Parameters Setting
    def __init__(self,id,name,fixed):
        self.id = id
        self.name = name
        self.mass = property[name]["mass"]
        self.radius = property[name]["radius"]
        self.type = property[name]["type"]
        self.fixed = fixed
        self.crash = False
        self.tracker = {"x":[],"y":[]}
    def set_coor_velocity(self,coor,velocity):
        self.coor = np.array(coor)
        self.velocity = np.array(velocity)
        self.update_boundary()

    #Basic Update
    def compute_dist_angle(self,obj):
        vec = obj.coor - self.coor
        angle = np.arctan2(vec[1],vec[0])
        dist = np.linalg.norm(vec)
        return angle,dist
    def compute_relative_velocity(self,obj):
        vec = self.velocity - obj.velocity
        return np.linalg.norm(vec)
    def update_boundary(self):
        self.boundary = {"x":[self.coor[0] + self.radius * np.cos(i) for i in np.linspace(0,2*np.pi,101,True)],
                         "y":[self.coor[1] + self.radius * np.sin(i) for i in np.linspace(0,2*np.pi,101,True)]}
    def update(self,obj_list):
        for i in obj_list:
            if i.id != self.id:
                angle,dist = self.compute_dist_angle(i)
                acc = g * i.mass / (dist**2)
                self.velocity += acc * np.array([np.cos(angle),np.sin(angle)])
                if dist <= i.radius:
                    self.crash = True
                    rel_vel = self.compute_relative_velocity(i)
                    print(f"\n{self.name} has collided with {i.name} at {rel_vel} m/s.")
        self.coor += self.velocity
    def update_tracker(self):
        self.tracker["x"].append(self.coor[0])
        self.tracker["y"].append(self.coor[1])
    
    #Setting and Update for spacecraft with propulsion
    def set_propulsion(self,orientation):
        self.fuel = property[self.name]["fuel"]
        self.mass_flow = property[self.name]["mass_flow"]
        self.thurst = property[self.name]["thurst"]
        self.rotate_max = property[self.name]["rotate_max"]
        self.orientation = orientation
    def control(self,thurst_percent,rotation):
        self.orientation += rotation
        if self.orientation < -np.pi:
            self.orientation += 2 * np.pi
        elif self.orientation > np.pi:
            self.orientation -= 2 * np.pi
        if self.fuel > 0:
            self.mass -= self.mass_flow * thurst_percent
            self.fuel -= self.mass_flow * thurst_percent
            if self.fuel < 0:
                print(f"\n{self.name} ran out of fuel.")
            acc = thurst_percent * self.thurst / self.mass
            self.velocity += acc * np.array([np.cos(self.orientation),np.sin(self.orientation)])

class simulator:
    def __init__(self,body_list,test_object):
        self.body_list = body_list
        self.test_object = test_object
        self.default_body_list = body_list
        self.default_test_object = test_object
    def step(self,angle_vel,thurst_percent = 1.0):
        for b in self.body_list:
            if not b.fixed and not b.crash:
                b.update(self.body_list)
                b.update_tracker()
        if not self.test_object.crash:
            self.test_object.control(thurst_percent,angle_vel)
            self.test_object.update(self.body_list)
            self.test_object.update_tracker()
    def plot_trajectory(self):
        plt.figure()
        for b in self.body_list:
            if not b.fixed:
                plt.plot(b.tracker["x"],b.tracker["y"],label=b.name)
            else:
                plt.plot(b.boundary["x"],b.boundary["y"],label=b.name)
        plt.plot(self.test_object.tracker["x"],self.test_object.tracker["y"],label=b.name)
        plt.legend()
        plt.show()