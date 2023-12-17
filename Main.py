import numpy as np
from scipy.optimize import minimize
import gym

if __name__ == "__main__":
    # Create the greenhouse environment
    env = GreenhouseEnvironment()

    # Generate expert trajectories using a heuristic controller
    expert_trajectories = []
    for _ in range(10):
        obs = env.reset()
        traj = []

        for _ in range(env.max_steps):
            action = expert_controller(obs)
            traj.append(action)
            obs, _, done, _ = env.step(action)

            if done:
                break

        expert_trajectories.append(traj)

    # Create and train the IRL agent
    irl_agent = IRLAgent(env)
    irl_agent.train(expert_trajectories)

    # Test the learned policy
    irl_agent.test_policy()