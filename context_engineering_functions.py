import pandas as pd
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
    rolling_df.reset_index(drop=True, inplace=True)
    rolling_df.columns = [x + '_is_available' for x in rolling_df.columns]

    df = pd.concat([df, rolling_df], axis=1)
    return df

def get_rewards(map_picks, demos):
    map_picks =  pd.merge(map_picks,
                          demos[['MatchId', 'MapName', 'WinnerId']], 
                          how = 'left', 
                          left_on= ['MatchId', 'MapName'], 
                          right_on = ['MatchId', 'MapName'],
                          suffixes = (None, '_right'))
    #for picks, if winner is same as picker, reward 1
    map_picks['Y_reward'] = (map_picks.DecisionTeamId == map_picks.WinnerId).astype(int)

    #drop extra WinnerId column since we no longer need it
    map_picks.dropna(inplace= True, axis = 0)
    map_picks.drop(labels = [ 'Created', 'Updated', 'WinnerId'], axis = 1, inplace = True)

    return map_picks


def create_basic_triples(data_directory, save = False):
    map_picks = pd.read_csv(os.path.join(data_directory, 'map_picks.csv'),
                            header = None, 
                            names = ['MatchId', 'MapName', 'DecisionOrder', 
                                      'DecisionTeamId', 'OtherTeamId',
                                      'Decision','Created', 'Updated'] )

    demos =  pd.read_csv(os.path.join(data_directory, 'demos.csv'),
                        header = None,
                        names = ['MatchId', 'MapName', 'WinnerId', 'WinnerScore', 'WinnerFirstHalfScore', 
                                  'WinnerSecondHalfScore', 'WinnerFirstHalfSide', 'WinnerOTScore', 'LoserId', 'LoserScore',
                                  'LoserFirstHalfScore', 'LoserSecondHalfScore', 'LoserFirstHalfSide', 'LoserOTScore',
                                'DemoParsed', 'Created', 'Updated'])


    map_encoder = {MapName: index for index, MapName in enumerate(sorted(map_picks.MapName.unique()))}

    map_pick_context = get_available_maps(map_picks)
    map_pick_context = map_pick_context[map_picks.Decision == 'Pick']

    map_pick_context.drop(labels = 'Decision', axis = 1, inplace = True)

    map_pick_context = get_rewards(map_pick_context, demos)


    map_pick_context.MapName = map_pick_context.MapName.map(map_encoder)

    map_pick_context.rename(columns = {'MapName': 'X_Action'}, inplace = True)



    cols = ['MatchId'] + \
          [i+'_is_available' for i in map_encoder.keys()] + \
           ['DecisionTeamId', 'OtherTeamId','DecisionOrder', 'X_Action','Y_reward' ]
    
    map_pick_context = map_pick_context[cols]
    

    if save:
        map_pick_context.to_csv(os.path.join(data_directory, 'basic_triples.csv'))
        print('Finished Basic Context Engineering')

    else:
        return map_pick_context





