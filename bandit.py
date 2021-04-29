#! /usr/bin/env python3

import numpy as np
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
        # theta shape: (n_features * n_arms,)
        self.theta = np.zeros((self.n_features * self.n_arms,))

    @property
    def current_baseline(self):
        """
        Return current baseline value.

        The baseline is just the running average of the rewards seen so far.

        input: None

        output: curr_baseline, current baseline. Shape: (1,)
        """
        return (self.reward_sum / self.iters
                if self.baseline and self.iters
                else 0)

    def _phi(self, X, a):
        """
        Return parameterization of X and action a.

        input: X, contexts. Shape: (1, n_features)
               a, actions. Shape: (1,)

        output: phi, parameterization. Shape: (n_features * n_arms)
        """
        if X.ndim == 1:
            X = X.reshape(1, -1)

        phi = np.zeros((self.n_arms, self.n_features))
        phi[a] = X

        return phi.reshape(-1, 1)

    def prefs(self, X):
        """
        Return the bandit's preferences for each action, for each context in X.

        input: X, contexts. Shape: (n_contexts, n_features)

        output: prefs, preferences. Shape: (n_contexts, n_arms)
        """

        # phis = np.stack([self._phi(X, a)
        #                  for a in range(self.n_arms)],
        #                 axis=1).squeeze().T

        prefs = np.array([self.theta.T @ self._phi(X, a)
                          for a in range(self.n_arms)]).T

        return prefs

    def predict_proba(self, X):
        """
        Return probabilities for each action for each context in X.

        input: X, contexts. Shape: (n_contexts, n_features)

        output: probs, probabilities per action. Shape: (n_contexts, n_arms)
        """

        prefs = self.prefs(X.reshape(-1, self.n_features))

        # use softmax to get probabilities from prefs
        probabilities = softmax(prefs.reshape(-1, self.n_arms),
                                axis=1)
        # get only possible maps
        # for now the context is just which maps are possible
        possible_maps = probabilities * X
        # re-normalize
        possible_maps /= possible_maps.sum()
        return possible_maps

    def predict(self, X, deterministic=True):
        """
        Return prediction of an action for each context in X.

        input: X, contexts. Shape: (n_contexts, n_features)

        output: action, action chosen per context. Shape: (n_contexts,)
        """

        predictions = self.predict_proba(X)

        # if we're deterministic, just pick the highest-probability action
        if deterministic:
            return predictions.argmax(axis=1)

        # otherwise, pick randomly from the distribution given by the probs
        # equivalent to using np.random.choice(),
        # but with per-row probabilities
        cumsum = predictions.cumsum(axis=1)
        random_val = np.random.random_sample(len(cumsum))
        binarized = (cumsum.T < random_val).astype(int).sum(axis=0)
        return binarized

    def _gradient(self, X, action):
        """
        Return the gradient of log of pi with respect to contexts X
        and actions A.

        input: X, contexts. Shape: (n_contexts, n_features)
               action, actions actually taken. Shape: (n_contexts,)

        output: grad, gradient of theta. Shape: (n_features,)
        """
        # should return \nabla log(pi(A_t|X_t))

        # precalc
        phis = [self._phi(X, i) for i in range(self.n_arms)]
        exps = [np.exp(self.theta.T @ phis[i])
                for i in range(self.n_arms)]
        phi = phis[action]

        numerator = sum([phis[i] * exps[i]
                         for i in range(self.n_arms)])
        denominator = sum(exps).sum()  # is an array

        return phi - numerator / denominator

    def update_theta(self, X, action, reward):
        """
        Update theta according to the context/action/reward triplets given.

        input: X, contexts. Shape: (n_contexts, n_features)
               action, actions actually taken. Shape: (n_contexts,)
               reward, rewards received for the actions. Shape: (n_contexts,)

        output: None
        """
        self.reward_sum += reward.sum()
        self.iters += len(X)
        r_t = reward.T - self.current_baseline
        gradient = r_t * self._gradient(X, action)
        self.theta += self.step_size * gradient.squeeze()
