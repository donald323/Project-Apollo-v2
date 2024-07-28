from stable_baselines3 import SAC
import earth_moon_simulator as ems

env = ems.earth_moon_env()

model = SAC("MlpPolicy",env,verbose=1)
model.learn(total_timesteps = int(1e9),log_interval = 1)
model.save("EMS_SAC")