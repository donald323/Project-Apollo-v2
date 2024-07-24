import numpy as np
import matplotlib.pyplot as plt
import tqdm
import json

g = 6.67e-11

with open("presets.json","r") as f:
    property = json.load(f)

class object:
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
    def update_boundary(self):
        self.boundary = {"x":[self.coor[0] + self.radius * np.cos(i) for i in np.linspace(0,2*np.pi,101,True)],
                         "y":[self.coor[1] + self.radius * np.sin(i) for i in np.linspace(0,2*np.pi,101,True)]}
    def compute_dist_angle(self,obj):
        vec = obj.coor - self.coor
        angle = np.arctan2(vec[1],vec[0])
        dist = np.linalg.norm(vec)
        return angle,dist
    def compute_relative_velocity(self,obj):
        vec = self.velocity - obj.velocity
        return np.linalg.norm(vec)
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
        self.update_boundary()
    def update_tracker(self):
        self.tracker["x"].append(self.coor[0])
        self.tracker["y"].append(self.coor[1])

class environment:
    def __init__(self,timestep,body_list):
        self.timestep = timestep
        self.body_list = body_list
    def simulate(self):
        for _ in tqdm.tqdm(range(self.timestep)):
            for o in self.body_list:
                if not o.fixed and not o.crash:
                    o.update(self.body_list)
                    o.update_tracker()
    def plot_trajectory(self):
        plt.figure()
        for o in self.body_list:
            if o.fixed:
                plt.plot(o.boundary["x"],o.boundary["y"],label=o.name)
            else:
                plt.plot(o.tracker["x"],o.tracker["y"],label=o.name)
        plt.legend()
        plt.show()

#Test Run

earth = object(id = 0,
               name = "Earth",
               fixed = True)
earth.set_coor_velocity(coor = [0.0,0.0],
                        velocity = [0.0,0.0])

moon_orbit = 3.84e8
moon = object(id = 1,
              name = "Moon",
              fixed = False)
moon.set_coor_velocity(coor = earth.coor + np.array([moon_orbit,0.0]),
                       velocity = np.array([0,np.sqrt(g*earth.mass/moon_orbit)]))

jilin1_orbit = moon_orbit * 0.75  
jilin1 = object(id = 2,
                name = "Jilin-1",
                fixed = False)
jilin1.set_coor_velocity(coor = earth.coor + np.array([jilin1_orbit,0.0]),
                         velocity =  np.array([0,np.sqrt(g*earth.mass/jilin1_orbit)]))

object_list = [earth, moon, jilin1]
env = environment(timestep = int(1e7),
                  body_list = object_list)
env.simulate()
env.plot_trajectory()