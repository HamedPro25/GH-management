import numpy as np
from scipy.optimize import minimize
import gym

class GreenhouseEnvironment(gym.Env):
    def __init__(self):
        super(GreenhouseEnvironment, self).__init__()

        self.observation_space = gym.spaces.Box(low=np.array([0.0, 0.0]), high=np.array([100.0, 100.0]), dtype=np.float32)
        self.action_space = gym.spaces.Discrete(9)  # Discrete actions representing different combinations of heating, cooling, and fan speed

        self.target_temperature = 22.0
        self.target_humidity = 60.0

        self.initial_temperature = 25.0
        self.initial_humidity = 50.0


        self.temperature = self.initial_temperature
        self.humidity = self.initial_humidity

    def reset(self):
        self.temperature = self.initial_temperature
        self.humidity = self.initial_humidity
        return np.array([self.temperature, self.humidity])

    def step(self, action):
        # Convert discrete action to continuous control signals
        heating, cooling, fan_speed = action // 3, (action % 3) - 1, (action % 3) - 1

        # Update temperature and humidity based on actions
        self.temperature += heating - cooling
        self.humidity += fan_speed * 2.0  # Simplified humidity control

        # Constrain values within reasonable ranges
        self.temperature = np.clip(self.temperature, 15.0, 30.0)
        self.humidity = np.clip(self.humidity, 30.0, 70.0)

        # Calculate reward: negative L1 distance to target temperature and humidity
        reward = -np.abs(self.temperature - self.target_temperature) - np.abs(self.humidity - self.target_humidity)


        # Return the next observation, reward, whether the episode is done, and additional information
        return np.array([self.temperature, self.humidity]), reward, {}

# Expert Controller (Heuristic)
def expert_controller(observation):
    target_temperature, target_humidity = observation
    heating, cooling, fan_speed = 0, 0, 0

    # Simple proportional control for temperature
    if target_temperature > observation[0]:
        heating = 1
    elif target_temperature < observation[0]:
        cooling = 1

    # Simple proportional control for humidity
    if target_humidity > observation[1]:
        fan_speed = 1
    elif target_humidity < observation[1]:
        fan_speed = -1

    return heating * 3 + cooling + fan_speed + 4  # Convert to discrete action
