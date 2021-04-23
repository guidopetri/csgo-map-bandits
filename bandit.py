#! /usr/bin/env python3

import numpy as np
import pandas as pd
from scipy.special import softmax


class Bandit(object):
    def __init__(self, n_features, n_arms, step_size=0.05, baseline=False):
        self.n_features = n_features
        self.n_arms = n_arms
        self.step_size = step_size
        self.baseline = baseline
        self.reward_sum = 0
        self.iters = 0

        # start at uniform
        self.theta = np.zeros((self.n_features, self.n_arms))
        
    @property
    def current_baseline(self):
        return (self.reward_sum / self.iters
                if self.baseline and self.iters
                else 0)
        
    def predict_proba(self, X):
        probabilities = softmax((X @ self.theta).reshape(-1, self.n_arms),
                                axis=1)
        possible_maps = probabilities * X
        possible_maps /= possible_maps.sum()
        return possible_maps
    
    def predict(self, X, deterministic=True):
        predictions = self.predict_proba(X)
        if deterministic:
            return predictions.argmax(axis=1)
        cumsum = predictions.cumsum(axis=1)
        random_val = np.random.random_sample(len(cumsum))
        binarized = (cumsum.T < random_val).astype(int).sum(axis=0)
        return binarized
    
    def theta_gradient(self, X, action):
        return np.eye(self.n_arms)[action].squeeze() - self.predict_proba(X)
    
    def update_theta(self, X, action, reward):
        self.reward_sum += reward.sum()
        self.iters += len(reward)
        r_t = reward.T - self.current_baseline
        gradient = r_t @ self.theta_gradient(X, action)
        self.theta[action] += self.step_size * gradient
