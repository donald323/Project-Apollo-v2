import numpy as np
import matplotlib.pyplot as plt
import tqdm
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
    def update(self,obj_list,ts):
        for i in obj_list:
            if i.id != self.id:
                angle,dist = self.compute_dist_angle(i)
                acc = g * i.mass / (dist**2)
                self.velocity += acc * np.array([np.cos(angle),np.sin(angle)])
                if dist <= i.radius:
                    self.crash = True
                    rel_vel = self.compute_relative_velocity(i)
                    print(f"\n{self.name} has collided with {i.name} at {rel_vel} m/s at {ts}s.")
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
    def propulsion(self,thurst_percent,rotation):
        self.orientation += rotation
        self.mass -= self.mass_flow * thurst_percent
        self.fuel -= self.mass_flow * thurst_percent
        acc = thurst_percent * self.thurst / self.mass
        self.velocity += acc * np.array([np.cos(self.orientation),np.sin(self.orientation)])

class environment:
    def __init__(self,body_list,test_object):
        self.body_list = body_list
        self.test_object = test_object
    def step(self,ts,angle_vel,thurst_percent = 1.0):
        for b in self.body_list:
            if not b.fixed and not b.crash:
                b.update(self.body_list,ts)
                b.update_tracker()
        if not self.test_object.crash:
            self.test_object.propulsion(thurst_percent,angle_vel)
            self.test_object.update(self.body_list,ts)
            self.test_object.update_tracker()
    def plot_trajectory(self):
        plt.figure()
        for b in self.body_list:
            if not b.fixed:
                plt.plot(b.tracker["x"],b.tracker["y"],label=b.name)
            else:
                plt.plot(b.boundary["x"],b.boundary["y"],label=b.name)
        plt.plot(self.test_object.tracker["x"],self.test_object.tracker["y"],label=b.name)
        plt.show()


#Test Run

#Earth
earth = object(id = 0,
               name = "Earth",
               fixed = True)
earth.set_coor_velocity(coor = [0,0],
                        velocity = [0,0])

#Moon
moon_orbit = 3.84e8
moon = object(id = 1,
              name = "Moon",
              fixed = False)
moon.set_coor_velocity(coor = [moon_orbit,0],
                       velocity = [0,compute_orbit_velocity(earth.mass,moon_orbit)])

#Lunar Explorer 2
le2_orbit = earth.radius + 4e5
le2 = object(id = 2,
             name = "Lunar-Explorer-2",
             fixed = False)
le2.set_coor_velocity(coor = [le2_orbit,0],
                      velocity = [0,compute_orbit_velocity(earth.mass,le2_orbit)])
le2.set_propulsion(0)

body_list = [earth,moon]
test_env = environment(body_list = body_list,
                       test_object = le2)
timestamps = int(1e6)

for ts in tqdm.tqdm(range(timestamps)):
    test_env.step(ts,-1e-3)

test_env.plot_trajectory()