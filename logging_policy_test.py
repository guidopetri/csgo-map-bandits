import pandas as pd
import numpy as np
from context_engineering_functions import *
from logging_policy import LoggingPolicy
import os

data_directory = '../data/clean/'
map_pick_context = create_basic_triples(data_directory)

# Compare p(a|x) for LoggingPolicy.get_pa_x(context) to manual calculation.
# Prints "Good" if the probabilities are equal

context_cols = ['de_dust2_is_available', 'de_inferno_is_available',
       'de_mirage_is_available', 'de_nuke_is_available',
       'de_overpass_is_available', 'de_train_is_available',
       'de_vertigo_is_available', 'DecisionTeamId', 'OtherTeamId',
       'DecisionOrder']
full_context = map_pick_context[context_cols]
full_action = map_pick_context['X_Action']

lp = LoggingPolicy(map_pick_context,map_pick_context['X_Action'])
context = full_context.loc[100]
full_pa_x = lp.pa_x_dict[context['DecisionTeamId']]

# Calculate probability distributions
from_lp = lp.predict_proba(context)[0]
manual = full_pa_x[0]/(1-full_pa_x[4]-full_pa_x[4])

print("predict_proba: ",from_lp)
print("manual calculation: ",manual)

if lp.predict_proba(context)[0] == manual:
    print("Good")
else:
	print("Test Failed")