import pandas as pd
import numpy as np

### Simpler Logging Policy
''' Approach:
- Only consider the DecisionTeamId - basically, the logging policy assumes that each team has a static Pick distribution that doesn't account for the rest of the context
- Zero out unavailable maps, reweight toward others
- Set a minimum probability of 1/num_samples for available maps so that pi/pi_0 is never zero.'''

class LoggingPolicy(object):
    ''' Logging Policy constructs p(a|x) where x is the DecisionTeamId.
    Init Inputs:
	- full_context: DataFrame containing the context features. Must contain the is_available columns for all 7 maps and DecisionTeamId. Other features are not used.
	- full_action: Series of ints corresponding to chosen map. Order should align with full_context 

	Use:
	- get_pa_x(context) returns p(a|x) accounting for unavailable maps. Should be used for Importance Weighting.
    '''
    def __init__(self, full_context, full_action):
        self.full_context = full_context
        self.full_action = full_action
        self.pa_x_dict = self.calculate_p_action_conditional()
        self.map_cols = ['de_dust2_is_available', 'de_inferno_is_available',
                         'de_mirage_is_available', 'de_nuke_is_available',
                         'de_overpass_is_available', 'de_train_is_available',
                         'de_vertigo_is_available']
    
    def calculate_p_action_conditional(self):
        # Calculate p(a|x) where x is the DecisionTeamId.
        # Sets minimum p(a|x) = 1/num_samples
        pa_x_dict = {}
        for team in self.full_context['DecisionTeamId'].unique():
            a_x = []
            for action in self.full_action.unique():
                a_x.append(max(1,(self.full_action[self.full_context['DecisionTeamId']==team]==action).sum()))
            pa_x = [ax/np.sum(a_x) for ax in a_x]
            pa_x_dict[team] = pa_x
        return pa_x_dict
    
    def predict_proba(self, X, is_veto=False):
        '''Function returns the p(a|x) where x is the DecisionTeamId, re-weighted to account for 
           unavailable maps.'''
        # Confirm that map_availability covers all maps
        assert pd.Series([m in X.index for m in self.map_cols]).all()

        # Zero out probability for unavailable maps, normalize the other probabilities
        pa_x = (self.pa_x_dict[X['DecisionTeamId']]).copy()
        for i,m in enumerate(self.map_cols):
            if X[m] == 0:
                pa_x[i] = 0
        pa_x = pa_x / np.sum(pa_x)
        if is_veto:
            pass
        else:
            return pa_x