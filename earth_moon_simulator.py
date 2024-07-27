import gymnasium as gym
import simulator as sim
import numpy as np
import utility as uti

gym.logger.set_level(40)

#Preset Parameters
def return_items():
    #Earth
    earth = sim.object(id = 0,
                name = "Earth",
                fixed = True)
    earth.set_coor_velocity(coor = [0,0],
                            velocity = [0,0])

    #Moon
    moon_orbit = 3.84e8
    moon = sim.object(id = 1,
                name = "Moon",
                fixed = False)
    moon.set_coor_velocity(coor = [moon_orbit,0],
                        velocity = [0,sim.compute_orbit_velocity(earth.mass,moon_orbit)])

    #Lunar Explorer 2
    le2_orbit = earth.radius + 4e5
    le2 = sim.object(id = 2,
                name = "Lunar-Explorer-2",
                fixed = False)
    le2.set_coor_velocity(coor = [le2_orbit,0],
                        velocity = [0,sim.compute_orbit_velocity(earth.mass,le2_orbit)])
    le2.set_propulsion(0)

    body_list = [earth,moon]
    return body_list, le2

class earth_moon_env(gym.Env):
    def __init__(self):
        super(earth_moon_env,self).__init__()
        body_list, test_object = return_items()
        self.simulator = sim.simulator(body_list,test_object)
        '''
        State Space
        - X and Y coordinate of the ship
        - X and Y coordinate of the Moon
        - Orientation of the ship
        - Remaining Fuel of the ship
        '''
        self.observation_space = gym.spaces.Box(low = np.array([-25.0,-25.0,-25.0,-25.0,-np.pi,0.0]),
                                                high = np.array([25.0,25.0,25.0,25.0,np.pi,self.simulator.test_object.fuel]),
                                                dtype = np.float32)
        '''
        Action Space
        - Angular Velocity
        - Thurst Percentage
        '''
        self.action_space = gym.spaces.Box(low = np.array([-0.1,0.0]),
                                          high = np.array([0.1,1.0]),
                                          dtype = np.float32)
        self.counter = 0
    def reset(self):
        body_list, test_object = return_items()
        self.simulator = sim.simulator(body_list,test_object)
        obs = np.array(
            [
                uti.log_encode(self.simulator.test_object.coor[0]),
                uti.log_encode(self.simulator.test_object.coor[1]),
                uti.log_encode(self.simulator.body_list[1].coor[0]),
                uti.log_encode(self.simulator.body_list[1].coor[1]),
                self.simulator.test_object.orientation,
                self.simulator.test_object.fuel
            ]
        )
        self.counter = 0
        return obs
    def compute_reward(self):
        '''
        Reward Scheme
        - Distance from moon (Difference from 100 km)
        - Crash?
        - Remaining Fuel Percentage
        '''
        _, dist = self.simulator.test_object.compute_dist_angle(self.simulator.body_list[1])
        r1 = np.exp(-1e-4 * dist)
        r2 = -100 if self.simulator.test_object.crash else 0
        r3 = self.simulator.test_object.fuel / 2500
        reward = r1 + r2 + r3
        return reward
    def step(self,action):
        rotate_vel = action[0]
        thurst_percent = action[1]
        self.simulator.step(rotate_vel,thurst_percent)
        next_obs = np.array(
            [
                uti.log_encode(self.simulator.test_object.coor[0]),
                uti.log_encode(self.simulator.test_object.coor[1]),
                uti.log_encode(self.simulator.body_list[1].coor[0]),
                uti.log_encode(self.simulator.body_list[1].coor[1]),
                self.simulator.test_object.orientation,
                self.simulator.test_object.fuel
            ],
            dtype = np.float32
        )
        reward = self.compute_reward()
        terminated = self.simulator.test_object.crash
        self.counter += 1
        truncated = self.counter > int(2e6)
        return next_obs, reward, terminated, truncated, {}