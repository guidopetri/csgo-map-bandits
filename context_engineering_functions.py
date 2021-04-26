import pandas as pd
import numpy as np
import os

def get_available_maps(df):
    df.sort_values(by=['MatchId', 'DecisionOrder'], ascending=True, inplace=True)
    df.reset_index(drop=True, inplace=True)
    
    map_names = df['MapName'].unique().tolist()

    # get which map we're talking about, OHE
    manip_df = pd.concat([df[['MatchId']], pd.get_dummies(df['MapName'])], axis=1)

    # 1 - what's not available
    rolling_df = (1 - (manip_df.groupby('MatchId')[map_names]
                               .rolling(7, min_periods=0)
                               .sum()  # rolling sum
                               .shift(1, fill_value=0)  # shift down so we know what's available
                               ))
    rolling_df = rolling_df.astype(int)
    rolling_df.reset_index(drop = True, inplace=True)
    rolling_df.columns = [x + '_is_available' for x in rolling_df.columns]

    df = pd.concat([df, rolling_df], axis=1)
    return df

def get_basic_rewards(map_picks, demos):
    map_picks =  pd.merge(map_picks,
                          demos[['MatchId', 
                                'MapName', 
                                'WinnerId',
                                'WinnerScore',
                                'LoserId', 
                                'LoserScore']], 
                          how = 'left', 
                          left_on = ['MatchId', 'MapName'], 
                          right_on = ['MatchId', 'MapName'],
                          suffixes = (None, '_right'))
    #for picks, if winner is same as picker, reward 1

    map_picks['MapWinnerId'] = np.where(map_picks.WinnerScore > map_picks.LoserScore, 
                                      map_picks.WinnerId, 
                                      map_picks.LoserId)

    map_picks['Y_reward'] = (map_picks.DecisionTeamId == map_picks.MapWinnerId).astype(int)

    #drop extra WinnerId column since we no longer need it
    map_picks.dropna(inplace= True, axis = 0)
    map_picks.drop(labels = ['WinnerId'], axis = 1, inplace = True)

    return map_picks

def get_proportion_rewards(map_picks, demos):
    map_picks =  pd.merge(map_picks,
                          demos[['MatchId', 
                                'MapName', 
                                'WinnerId',
                                'WinnerScore',
                                'LoserId', 
                                'LoserScore']], 
                          how = 'left', 
                          left_on = ['MatchId', 'MapName'], 
                          right_on = ['MatchId', 'MapName'],
                          suffixes = (None, '_right'))

    map_picks['MapWinnerId'] = np.where(map_picks.WinnerScore > map_picks.LoserScore, 
                                      map_picks.WinnerId, 
                                      map_picks.LoserId)

    #proportion of matches won
    map_picks['Y_reward'] = np.where(map_picks.MapWinnerId == map_picks.DecisionTeamId,
                                    (map_picks.WinnerScore - map_picks.LoserScore ) / (map_picks.WinnerScore + map_picks.LoserScore),
                                    (map_picks.LoserScore - map_picks.WinnerScore ) / (map_picks.WinnerScore + map_picks.LoserScore))             

    map_picks.dropna(inplace= True, axis = 0)

    return map_picks


def create_basic_triples(data_directory,reward_function = get_basic_rewards, save = False):
    map_picks = pd.read_csv(os.path.join(data_directory, 'map_picks.csv'))

    demos =  pd.read_csv(os.path.join(data_directory, 'demos.csv'))

    map_encoder = {MapName: index for index, MapName in enumerate(sorted(map_picks.MapName.unique()))}

    map_pick_context = get_available_maps(map_picks)
    
    map_pick_context = map_pick_context[map_picks.Decision == 'Pick']

    map_pick_context.drop(labels = 'Decision', axis = 1, inplace = True)

    map_pick_context = reward_function(map_pick_context, demos)

    map_pick_context.MapName = map_pick_context.MapName.map(map_encoder)

    map_pick_context.rename(columns = {'MapName': 'X_Action'}, inplace = True)

    cols = ['MatchId'] + \
          [i+'_is_available' for i in map_encoder.keys()] + \
          ['DecisionTeamId', 'OtherTeamId','DecisionOrder', 'X_Action','Y_reward']
    
    map_pick_context = map_pick_context[cols]
    print('Finished Basic Context Engineering')
    if save:
        map_pick_context.to_csv(os.path.join(data_directory, 'basic_triples.csv'))
        return map_pick_context

    else:
        return map_pick_context





