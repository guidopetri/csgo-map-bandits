import numpy as np
import pandas as pd
from bandit import Bandit
from logging_policy import LoggingPolicy
from sklearn.linear_model import RidgeCV

def train_value_estimator(context_train,map_picks_train,actions_train,rewards_train,log_policy,target_bandit,veto_flags=None):
    '''
    Trains an importance weighted RidgeCV model which is used for direct method estimation.
    
    Input:
        context_train (np.array(n x k)): n is number of train examples. k is dimensionality of context.
        map_picks_train (df)
            map_picks train and context train must be created using this line of code:
            "context_train = map_pick_context_train[cols].values"

        actions_train (np.array(n)) : actions taken
        rewards (np.array(n): rewards received
        log_policy: a LoggingPolicy object, which needs to have a function predict_proba(self,context)
        target_policy: a Bandit object, which needs to have a function predict_proba(self,context)
    '''
    all_actions = np.unique(actions_train) # return from unique is already sorted
    action_to_model_dict = {}
    log_propensities = np.empty((context_train.shape[0],len(log_policy.map_cols)))

    # Veto flags
    if veto_flags is not None:
        veto_flags = (veto_flags == 'veto')
    else:
        veto_flags = pd.Series([False]*context_train.shape[0])
    
    for ii,(idx,row) in enumerate(map_picks_train.iterrows()):   
        log_propensities_row = log_policy.predict_proba(row,veto_flags.iloc[ii])
        log_propensities[ii,:] = log_propensities_row
    
    target_propensities = np.empty((context_train.shape[0],len(log_policy.map_cols)))
    
    for ii in range(context_train.shape[0]):   
        target_propensities_row = target_bandit.predict_proba(context_train[ii,:], veto_flags.iloc[ii])
        target_propensities[ii,:] = target_propensities_row
    
    # Check to make sure these are both n x k. I guess its possible that not all actions were chosen, but thats unlikely.
    assert log_propensities.shape == target_propensities.shape
    assert log_propensities.shape == (context_train.shape[0], len(all_actions))

    
    #Fit a model for each action
    for action in all_actions:
        context_for_action = context_train[actions_train==action,:]
        rewards_for_action = rewards_train[actions_train==action]
        model = RidgeCV()
        t_prop_action = target_propensities[actions_train==action,action]
        l_prop_action = log_propensities[actions_train==action,action]
        importance_weights = np.divide(t_prop_action, l_prop_action, out=np.zeros_like(t_prop_action), where=l_prop_action!=0)
        model.fit(context_for_action,rewards_for_action,sample_weight=importance_weights )
        action_to_model_dict[action] = model
    # Models are fit
    return action_to_model_dict

def evaluate(context_test,map_picks_test,actions_test,rewards_test,log_policy,target_bandit,action_to_model_dict,veto_flags=None):
    est = {}
    est["mean"] = np.mean(rewards_test)
    
    all_actions = action_to_model_dict.keys()
    num_actions = target_bandit.n_arms
    assert target_bandit.n_arms == len(log_policy.pa_x_dict[6])

    # Veto flags
    if veto_flags is not None:
        veto_flags = (veto_flags == 'veto')
    else:
        veto_flags = pd.Series([False]*context_test.shape[0])
    
    #Create Logging policies propensity distribution
    log_propensities = np.empty((context_test.shape[0],len(log_policy.map_cols)))
    for ii,(idx,row) in enumerate(map_picks_test.iterrows()):  
        log_propensities_row = log_policy.predict_proba(row,veto_flags.iloc[ii])
        log_propensities[ii,:] = log_propensities_row
    
    #Create target policies propensity distribution
    target_propensities = np.empty((context_test.shape[0],num_actions))
    for ii in range(context_test.shape[0]):   
        target_propensities_row = target_bandit.predict_proba(context_test[ii,:], veto_flags.iloc[ii])
        target_propensities[ii,:] = target_propensities_row
   
    #( Self-normalized) Importance weighted value estimator
    
    importance_weights_matrix = np.divide(target_propensities,log_propensities,out=np.zeros_like(target_propensities), where=log_propensities!=0)
    importance_weights = np.choose(actions_test,importance_weights_matrix.T)

    est['IW'] = (rewards_test * importance_weights).mean()
    
    #SN IW estimator
    est['SN_IW'] = (rewards_test*importance_weights).sum()/importance_weights.sum()
    
    #Direct Method
    
    predicted_rewards = np.empty((context_test.shape[0],num_actions))
    #Create predicted reward distribution
    for action in all_actions:
        model = action_to_model_dict[action]
        predicted_rewards[:,action] = model.predict(context_test)
    # estimate
    est['Direct_Method_IW'] = (predicted_rewards*target_propensities).sum(axis=1).mean()
    return est
    