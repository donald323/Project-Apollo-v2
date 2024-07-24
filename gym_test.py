import gymnasium as gym

env = gym.make(
    id = "LunarLander-v2",
    render_mode = "human",
    continuous = True,
    gravity = -10,
    enable_wind = False,
    wind_power = 15.0,
    turbulence_power = 1.5
)

from stable_baselines3 import SAC

model = SAC("MlpPolicy",env,verbose=1)
model.learn(total_timesteps=1000000)
model.save("lunarlander_dqn")