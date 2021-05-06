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

    def prefs(self, X, action_type='pick'):
        """
        Return the bandit's preferences for each action, for each context in X.

        input: X, contexts. Shape: (n_contexts, n_features)

        output: prefs, preferences. Shape: (n_contexts, n_arms)
        """

        # phis = np.stack([self._phi(X, a)
        #                  for a in range(self.n_arms)],
        #                 axis=1).squeeze().T

        if (action_type == 'pick') | (len(self.theta) == self.n_features*self.n_arms):
            theta_slice = self.theta[:self.n_features * self.n_arms]
        elif (action_type == 'veto') & (len(self.theta) != self.n_features*self.n_arms):
            theta_slice = self.theta[self.n_features * self.n_arms:]
        else:
            raise ValueError(f"wrong action_type passed: {action_type}")

        prefs = np.array([theta_slice.T @ self._phi(X, a)
                          for a in range(self.n_arms)]).T

        return prefs

    def predict_proba(self, X, action_type='pick'):
        """
        Return probabilities for each action for each context in X.

        input: X, contexts. Shape: (n_contexts, n_features)

        output: probs, probabilities per action. Shape: (n_contexts, n_arms)
        """

        prefs = self.prefs(X.reshape(-1, self.n_features), action_type)

        # use softmax to get probabilities from prefs
        probabilities = softmax(prefs.reshape(-1, self.n_arms),
                                axis=1)
        # get only possible maps
        # for now the context is just which maps are possible
        possible_maps = probabilities * X[:self.n_arms]
        # re-normalize
        possible_maps /= possible_maps.sum()
        return possible_maps

    def predict(self, X, deterministic=True, action_type='pick'):
        """
        Return prediction of an action for each context in X.

        input: X, contexts. Shape: (n_contexts, n_features)

        output: action, action chosen per context. Shape: (n_contexts,)
        """

        predictions = self.predict_proba(X, action_type)

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

    def _gradient(self, X, action, action_type='pick'):
        """
        Return the gradient of log of pi with respect to contexts X
        and actions A.

        input: X, contexts. Shape: (n_contexts, n_features)
               action, actions actually taken. Shape: (n_contexts,)

        output: grad, gradient of theta. Shape: (n_features,)
        """
        # should return \nabla log(pi(A_t|X_t))

        if action_type == 'pick':
            theta_slice = self.theta[:self.n_features * self.n_arms]
        elif action_type == 'veto':
            theta_slice = self.theta[self.n_features * self.n_arms:]
        else:
            raise ValueError(f"wrong action_type passed: {action_type}")

        # precalc
        phis = [self._phi(X, i) for i in range(self.n_arms)]
        exps = [np.exp(theta_slice.T @ phis[i])
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

class BothBandit(Bandit):
    def __init__(self, n_features, n_arms, step_size=0.05, baseline=False, trained_type='pick'):
        self.n_features = n_features
        self.n_arms = n_arms
        self.step_size = step_size
        self.baseline = baseline
        self.reward_sum = 0
        self.iters = 0
        self.trained_type = trained_type

        # start at uniform
        # theta shape: (n_features * n_arms,)
        self.theta = np.zeros((self.n_features * self.n_arms,))

    def prefs(self, X, action_type='pick'):
        """
        Return the bandit's preferences for each action, for each context in X.

        input: X, contexts. Shape: (n_contexts, n_features)
                action_type. 'pick' or 'veto'

        output: prefs, preferences. Shape: (n_contexts, n_arms)
        """

        # phis = np.stack([self._phi(X, a)
        #                  for a in range(self.n_arms)],
        #                 axis=1).squeeze().T
        theta_slice = self.theta[:self.n_features * self.n_arms]
        prefs = np.array([theta_slice.T @ self._phi(X, a)
                          for a in range(self.n_arms)]).T
        if action_type == self.trained_type:
            prefs_out = prefs 
        else:
            prefs_out  = (1-prefs)/((1-prefs).sum())
        
        return prefs_out

class VetoBandit(Bandit):
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
        gradient = np.zeros(self.theta.shape)
        for idx, x in enumerate(X):
            gradient += r_t[idx] * self._gradient(x, action[idx]).squeeze()
        self.theta += self.step_size * gradient.squeeze()


class ComboBandit(Bandit):
    def __init__(self, n_features, n_arms, step_size=0.05, baseline=False):
        self.n_features = n_features
        self.n_arms = n_arms
        self.step_size = step_size
        self.baseline = baseline
        self.reward_sum = 0
        self.iters = 0

        # start at uniform
        # theta shape: (2 * n_features * n_arms,)
        # where one set is for picks and another for vetos
        self.theta = np.zeros((2 * self.n_features * self.n_arms,))

    def update_theta(self, X, action, reward, action_type):
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
        gradient = np.zeros(self.theta.shape)
        half_size = self.n_features * self.n_arms

        # allow both episodic and online learning
        if X.ndim == 1:
            X = X.reshape(-1, 1)

        for idx, x in enumerate(X):
            nabla = self._gradient(x, action[idx], action_type).squeeze()
            if action_type == 'pick':
                gradient[:half_size] += r_t[idx] * nabla
            elif action_type == 'veto':
                # add 7 to offset the vetos
                # nabla = self._gradient(x, action[idx] + self.n_arms).squeeze()
                gradient[half_size:] += r_t[idx] * nabla
            else:
                raise ValueError('Action type must be one of "pick", "veto"')

        self.theta += self.step_size * gradient.squeeze()


class EpisodicBandit(ComboBandit):
    def update_theta(self, X, action, reward, action_types):
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
        gradient = np.zeros(self.theta.shape)
        half_size = self.n_features * self.n_arms

        # enforce episodic learning
        if X.ndim == 1:
            raise ValueError("Only episodic learning intended: "
                             "X must be 2-dimensional")

        for idx, x in enumerate(X):
            nabla = self._gradient(x, action[idx], action_types[idx]).squeeze()
            if action_types[idx] == 'pick':
                gradient[:half_size] += r_t[idx] * nabla
            elif action_types[idx] == 'veto':
                # add 7 to offset the vetos
                # nabla = self._gradient(x, action[idx] + self.n_arms).squeeze()
                gradient[half_size:] += r_t[idx] * nabla
            else:
                raise ValueError('Action type must be one of "pick", "veto"')

        self.theta += self.step_size * gradient.squeeze()
