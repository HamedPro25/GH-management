import numpy as np
from scipy.optimize import minimize
import gym

class IRLAgent:
    def __init__(self, env):
        self.env = env

    def feature_expectations(self, traj, gamma=0.99):
        """
        Calculate feature expectations from a trajectory.
        """
        T = len(traj)
        feat_exp = np.zeros(self.env.action_space.n)

        for t in range(T):
            action = traj[t]
            feat_exp[action] += gamma ** t

        return feat_exp / T

    def reward_function(self, weights, action):
        """
        Linear reward function.
        """
        return np.dot(weights, self.env.feature_vector(action))

    def maxent_irl(self, expert_trajectories, gamma=0.99, learning_rate=0.01, n_iterations=100):
        """
        Maximum Entropy Inverse Reinforcement Learning.
        """
        # Initialize weights randomly
        weights = np.random.rand(self.env.action_space.n)

        for _ in range(n_iterations):
            # Compute feature expectations from expert trajectories
            expert_feat_exp = np.zeros(self.env.action_space.n)
            for traj in expert_trajectories:
                expert_feat_exp += self.feature_expectations(traj, gamma)

            # Compute feature expectations from current policy
            policy_feat_exp = np.zeros(self.env.action_space.n)
            for traj in self.generate_trajectories(weights, gamma):
                policy_feat_exp += self.feature_expectations(traj, gamma)

            # Update weights using gradient ascent
            weights += learning_rate * (expert_feat_exp - policy_feat_exp)

        return weights

    def generate_trajectories(self, weights, gamma=0.99, n_trajectories=10, max_steps=100):
        """
        Generate trajectories using the current policy.
        """
        trajs = []

        for _ in range(n_trajectories):
            obs = self.env.reset()
            traj = []

            for _ in range(max_steps):
                action = self.get_action(weights, obs)
                traj.append(action)
                obs, _, done, _ = self.env.step(action)

                if done:
                    break

            trajs.append(traj)

        return trajs

    def get_action(self, weights, observation):
        """
        Get action using the softmax policy.
        """
        logits = np.dot(weights, self.env.feature_vector(observation))
        action_probs = np.exp(logits) / np.sum(np.exp(logits))
        return np.random.choice(self.env.action_space.n, p=action_probs)

    def train(self, expert_trajectories):
        self.weights = self.maxent_irl(expert_trajectories)

    def test_policy(self, num_episodes=10):
        for _ in range(num_episodes):
            obs = self.env.reset()
            total_reward = 0

            while True:
                action = self.get_action(self.weights, obs)
                obs, reward, done, _ = self.env.step(action)
                total_reward += reward

                if done:
                    break

            print(f"Total Reward: {total_reward}")