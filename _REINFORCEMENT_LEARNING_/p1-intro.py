import gym
import gymnasium
from stable_baselines3 import A2C


env = gym("CartPole-v1")
env.reset()


model = A2C("MlpPolicy", env, verbose=1)
model.learn(total_timesteps = 10000)

# print("sample action : ", env.action_space.sample())
# print("observation space shape: ", env.observation_space.shape)
# print("sample observation : ", env.observation_space.sample())

episodes = 10

for ep in range(episodes):
    obs = env.reset()
    done = False
    while not done:
        env.render()
        obs, reward, done, info = env.step(env.action_space.sample())
        #print(info)
    
env.close()