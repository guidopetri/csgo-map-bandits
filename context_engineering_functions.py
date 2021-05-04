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
                               .reset_index(level = 0)
                               .groupby('MatchId')
                               .shift(1, fill_value = 0)# shift down so we know what's available
                               ))
    rolling_df.drop('MatchId',axis = 1, inplace = True)

    rolling_df = rolling_df.astype(int)
    rolling_df.reset_index(drop = True, inplace=True)
    rolling_df.columns = [x + '_is_available' for x in rolling_df.columns]

    df = pd.concat([df, rolling_df], axis=1)
    return df

def get_historical_win_pct(map_picks, matches, alpha = 5, beta = 10):
    matches = matches[['MatchId', 'MatchDate', 'MatchTime', 'WinnerId', 'LoserId']].copy()
    
    matches.sort_values(by = ['MatchDate', 'MatchTime'], inplace = True)
    matches.reset_index(drop = True, inplace = True)

    matches['Team_Win_Count'] = matches.groupby(by = 'WinnerId')[['WinnerId']].cumcount() 

    matches_long = pd.melt(matches, 
                          id_vars = ['MatchId','MatchDate','MatchTime'], 
                          value_vars = ['WinnerId','LoserId'], 
                          var_name = 'isWinner', 
                          value_name = 'TeamId')

    matches_long.sort_values(by = ['MatchDate','MatchTime'], inplace= True)
    
    matches_long['NumGames'] = matches_long.groupby(by = 'TeamId').cumcount() + 1
    
    matches_long.isWinner = (matches_long.isWinner == 'WinnerId').astype(int)
    
    matches_long['Team_Win_Count'] = matches_long.groupby('TeamId').isWinner.cumsum()
    
    matches_long['Team_Win_Rate'] = (matches_long.Team_Win_Count + alpha) / (matches_long.NumGames + beta)

    matches_long.Team_Win_Rate =  matches_long.groupby('TeamId').Team_Win_Rate.shift(1, fill_value = 0.5)

    matches_pivot = pd.pivot(matches_long, 
                            index = ['MatchId'],
                            columns = 'isWinner', 
                            values = ['TeamId','Team_Win_Rate'])

    matches_pivot.columns = ['LoserId', 'WinnerId', 'LoserWinPercentage', 'WinnerWinPercentage']

    map_picks = pd.merge(map_picks, 
                        matches_pivot, 
                        how = 'left', 
                        left_on = ['MatchId'],
                        right_on = ['MatchId'])

    map_picks['DecisionTeam_WinPercent'] = np.where(map_picks.DecisionTeamId == map_picks.WinnerId, 
                                                  map_picks.WinnerWinPercentage,
                                                  map_picks.LoserWinPercentage)
    
    map_picks['OtherTeam_WinPercent'] = np.where(map_picks.OtherTeamId == map_picks.WinnerId,
                                                map_picks.WinnerWinPercentage, 
                                                map_picks.LoserWinPercentage)

    map_picks.drop(labels = ['LoserId', 'WinnerId', 'LoserWinPercentage', 'WinnerWinPercentage'], axis = 1, inplace = True)

    return map_picks

def get_historical_map_win_pct(map_picks, demos, matches, alpha = 5, beta = 10):

    demos = demos.merge(matches[['MatchId', 'MatchDate', 'MatchTime'] ], on='MatchId')
    map_picks = map_picks.merge(matches[['MatchId', 'MatchDate', 'MatchTime'] ], on='MatchId')
    map_names = sorted(map_picks.MapName.unique())

    demos['MapWinner'] = np.where(demos.WinnerScore > demos.LoserScore, demos.WinnerId, demos.LoserId) # Mark which side won

    long_df = pd.melt(demos,
                    id_vars=['MatchId','MapName','MapWinner','MatchDate','MatchTime'],
                    value_vars = ['WinnerId','LoserId'],
                    var_name='Side',
                    value_name='TeamId').sort_values(by=['MatchDate', 'MatchTime'])

    long_df['MapResult'] = (long_df.MapWinner == long_df.TeamId).astype(int)

    long_df['MatchesOnMap'] = long_df.groupby(['MapName','TeamId'])['MapResult'].cumcount() + 1
    long_df['WinsOnMap'] = long_df.groupby(['MapName','TeamId'])['MapResult'].cumsum()

    # Add Laplace Smoothing
    long_df.WinsOnMap += alpha
    long_df.MatchesOnMap += beta
    long_df['WinRateOnMap'] = long_df.WinsOnMap/long_df.MatchesOnMap
  
    long_df['PrevWinRateOnMap'] = long_df.groupby(['MapName','TeamId']).WinRateOnMap.shift(1, fill_value = alpha/beta)

    pivot_df = pd.pivot(long_df, 
                       index = ['MatchId', 'MapName'],
                       columns = 'Side', 
                       values = ['TeamId','PrevWinRateOnMap'])

    pivot_df.columns = ['LoserId', 'WinnerId', 'LoserMap_HistWinPct', 'WinnerMap_HistWinPct']

    pivot_df.reset_index(inplace = True)

    map_picks = pd.merge(map_picks, 
                        pivot_df, 
                        how = 'left', 
                        left_on = ['MatchId', 'MapName'],
                        right_on = ['MatchId', 'MapName'])

    map_picks.sort_values(by = ['MatchDate', 'MatchTime'], inplace = True)

    map_picks.LoserId = map_picks.groupby('MatchId').LoserId.ffill().bfill()
    map_picks.WinnerId = map_picks.groupby('MatchId').WinnerId.ffill().bfill()

    #first forward fill Historical Map Win Percents, and then backfill all initials to alpha/beta
    map_picks.WinnerMap_HistWinPct = map_picks.groupby(by = ['WinnerId', 'MapName']).WinnerMap_HistWinPct.fillna(method = 'ffill')
    map_picks.WinnerMap_HistWinPct = map_picks.groupby(by = ['WinnerId', 'MapName']).WinnerMap_HistWinPct.fillna(value = alpha/beta)

    map_picks.LoserMap_HistWinPct = map_picks.groupby(by = ['LoserId', 'MapName']).LoserMap_HistWinPct.fillna(method = 'ffill')
    map_picks.LoserMap_HistWinPct = map_picks.groupby(by = ['LoserId', 'MapName']).LoserMap_HistWinPct.fillna(value = alpha/beta)


    map_picks['DecisionTeam_HistMapWinPercent'] = np.where(map_picks.DecisionTeamId == map_picks.WinnerId, 
                                                  map_picks.WinnerMap_HistWinPct,
                                                  map_picks.LoserMap_HistWinPct)
    
    map_picks['OtherTeam_HistMapWinPercent'] = np.where(map_picks.OtherTeamId == map_picks.WinnerId,
                                                map_picks.WinnerMap_HistWinPct, 
                                                map_picks.LoserMap_HistWinPct)

    pivot_table2 = pd.pivot(map_picks,
                          index = ['MatchId'],
                          columns = 'MapName', 
                          values = ['DecisionTeam_HistMapWinPercent','OtherTeam_HistMapWinPercent'])

    map_picks.drop(labels = ['LoserId', 'WinnerId', 
                            'MatchDate', 'MatchTime',
                            'WinnerMap_HistWinPct', 'LoserMap_HistWinPct',
                            'DecisionTeam_HistMapWinPercent','OtherTeam_HistMapWinPercent' ], axis = 1, inplace = True)

    pivot_table2.columns = [f'DecTeam_{i}_WinPct' for i in map_names] + [f'OtherTeam_{i}_WinPct' for i in map_names]
    pivot_table2.reset_index(inplace = True)

    map_picks = map_picks.merge(pivot_table2, how = 'left', left_on = 'MatchId', right_on = 'MatchId')
    map_picks.sort_values(by = ['MatchId', 'DecisionOrder'], inplace = True)

    return map_picks

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

def get_veto_rewards(map_picks, demos):
      map_picks_vetos = map_picks[map_picks.Decision == 'Remove'].sort_values(by = ['MatchId', 'DecisionOrder'])
      match_winners = demos[['MatchId', 'WinnerId']].drop_duplicates()

      map_picks_vetos =  pd.merge(map_picks_vetos,
                          match_winners, 
                          how = 'left', 
                          left_on = 'MatchId', 
                          right_on = 'MatchId',
                          suffixes = (None, '_right'))

      map_picks_vetos['number_to_exp'] = map_picks_vetos.DecisionOrder.replace(to_replace = {5:3, 6:4})

      map_picks_vetos['Y_reward'] = np.where(map_picks_vetos.WinnerId ==map_picks_vetos.DecisionTeamId, 
                                    1 , -1)/2**(map_picks_vetos.number_to_exp)

      map_picks_vetos.drop(labels = ['number_to_exp'], axis = 1, inplace = True)

      return map_picks_vetos

def create_basic_triples(data_directory,reward_function = get_basic_rewards, save = False):
    map_picks = pd.read_csv(os.path.join(data_directory, 'map_picks.csv'))

    demos =  pd.read_csv(os.path.join(data_directory, 'demos.csv'))

    map_encoder = {MapName: index for index, MapName in enumerate(sorted(map_picks.MapName.unique()))}

    map_pick_context = get_available_maps(map_picks)
    
    map_pick_context = reward_function(map_pick_context, demos)

    map_pick_context.drop(labels = 'Decision', axis = 1, inplace = True)

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

def create_basic_pick_veto_triples(data_directory,
                                  pick_reward_function = get_basic_rewards, 
                                  veto_reward_function = get_veto_rewards, 
                                  concat = False,
                                  save = False):
    
    map_picks = pd.read_csv(os.path.join(data_directory, 'map_picks.csv'))

    demos =  pd.read_csv(os.path.join(data_directory, 'demos.csv'))

    matches = pd.read_csv(os.path.join(data_directory, 'matches.csv'))

    map_encoder = {MapName: index for index, MapName in enumerate(sorted(map_picks.MapName.unique()))}

    map_pick_context = get_available_maps(map_picks)

    map_pick_context = get_historical_win_pct(map_pick_context, matches)
    map_pick_context = get_historical_map_win_pct(map_pick_context, demos, matches)

    rewards_list  = [pick_reward_function(map_pick_context, demos), veto_reward_function(map_pick_context, demos)]

    cols = ['MatchId'] + \
          [i+'_is_available' for i in map_encoder.keys()] + \
          ['DecisionTeamId', 'OtherTeamId', 'DecisionTeam_WinPercent', 'OtherTeam_WinPercent'] + \
          [f'DecTeam_{i}_WinPct' for i in map_encoder.keys()] + \
          [f'OtherTeam_{i}_WinPct' for i in map_encoder.keys()] + \
          ['DecisionOrder', 'MapName','Y_reward']

    for i in range(len(rewards_list)):
        rewards_list[i].drop(labels = 'Decision', axis = 1, inplace = True)
        rewards_list[i] = rewards_list[i][cols]
        rewards_list[i].rename(columns = {'MapName': 'X_Action'}, inplace = True)
        rewards_list[i].X_Action = rewards_list[i].X_Action.map(map_encoder)

    if concat:
        
        map_pick_context = pd.concat(rewards_list, axis = 0).sort_values(by = ['MatchId', 'DecisionOrder'])
        if save:
            map_pick_context.to_csv(os.path.join(data_directory, 'pick_veto_reward_triples.csv'))
        
        return map_pick_context

    else:
        if save:
            rewards_list[0].to_csv(os.path.join(data_directory, 'pick_reward_triples.csv'))
            rewards_list[1].to_csv(os.path.join(data_directory, 'veto_reward_triples.csv'))

        return tuple(rewards_list)


