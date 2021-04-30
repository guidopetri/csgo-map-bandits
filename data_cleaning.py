import os

import numpy as np
import pandas as pd

cols_demos = [
    'MatchId',
    'MapName' ,
    'WinnerId' ,
    'WinnerScore' ,
    'WinnerFirstHalfScore' ,
    'WinnerSecondHalfScore' ,
    'WinnerFirstHalfSide' ,
    'WinnerOTScore' ,
    'LoserId' ,
    'LoserScore' ,
    'LoserFirstHalfScore' ,
    'LoserSecondHalfScore' ,
    'LoserFirstHalfSide' ,
    'LoserOTScore' ,
    'DemoParsed' ,
    'Created' ,
    'Updated']
cols_map_picks = [
    'MatchId',
    'MapName',
    'DecisionOrder',
    'DecisionTeamId',
    'OtherTeamId',
    'Decision',
    'Created',
    'Updated']
cols_matches = [
    'MatchId',
    'HLTVMatchId',
    'CompetitionId',
    'HLTVLink',
    'MatchType',
    'MatchDate',
    'MatchTime',
    'Stars',
    'Slug',
    'WinnerId',
    'WinnerScore',
    'LoserId',
    'LoserScore',
    'Created',
    'Updated']
cols_teams = [
    'TeamId',
    'HLTVTeamId',
    'HLTVLink',
    'TeamName',
    'Country',
    'Twitter',
    'Facebook',
    'Created',
    'Updated']
cols_player_demos = [
    'MatchId',
    'PlayerId',
    'TeamId',
    'MapName',
    'Side',
    'Kills',
    'Deaths',
    'ADR',
    'KAST',
    'HLTVRating',
    'Created',
    'Updated']
cols_players = [
    'PlayerId',
    'HLTVPlayerId',
    'HLTVLink',
    'Country',
    'RealName',
    'PlayerName',
    'Facebook',
    'Twitter',
    'Twitch',
    'Created',
    'Updated']
data_dir = '../data/'
demos = pd.read_csv(data_dir+'demos.csv',names = cols_demos)
map_picks = pd.read_csv(data_dir+'map_picks.csv',names = cols_map_picks)
matches = pd.read_csv(data_dir+'matches.csv',names = cols_matches)
teams = pd.read_csv(data_dir+'teams.csv',names = cols_teams)
players = pd.read_csv(data_dir+'players.csv',names = cols_players)
player_demos = pd.read_csv(data_dir+'player_demos.csv',names = cols_player_demos)


def build_team_record_df(teams,demos):
    '''Creates team_record dataframe where each row is a Team, columns are various stats.
       Inputs are dataframes of the .csvs of the same name.
       Output is a dataframe indexed by teamid'''
    # All team IDs
    team_record = pd.DataFrame(teams['TeamId'].unique().astype(int),columns=['TeamId'])
    # Wins and losses (For games, not matches) by team ID
    game_wins = demos['WinnerId'].value_counts()
    game_losses = demos['LoserId'].value_counts()
    # merge wins and losses 
    team_record = team_record.merge(game_wins,left_on='TeamId',right_index=True,how='left').fillna(0).astype(int)
    team_record = team_record.merge(game_losses,left_on='TeamId',right_index=True,how='left').fillna(0).astype(int)
    team_record.rename(columns={"WinnerId": "GameWins", "LoserId": "GameLosses"},inplace=True)
    team_record['TotalGames'] = team_record['GameWins']+team_record['GameLosses'].astype(int)
    team_record['WinPercent'] = (team_record['GameWins']/team_record['TotalGames']).fillna(0)
    # Set index to team_id, to simplify things
    team_record.set_index('TeamId',drop=False,inplace=True)
    return team_record    


def remove_records(df,cols,remove_ids):
    # Function removes rows where df[cols] value is in remove_ids
    # Returns df with same columns and fewer rows

    few_games_dict = {tid: [] for tid in cols}
    out = [False]*df.shape[0]
    for col in cols:
        a = [t in remove_ids for t in df[col]]
        out = [a or b for (a,b) in zip(out,a)]
    df['remove'] = out
    print("Num removed: ",df['remove'].sum())
    # Remove bad records
    df_out = df[df['remove']==False]
    df_out = df_out.drop(['remove'],axis=1)

    return df_out


### Matches without map_picks data
# 26 maps in matches.csv do not have map_pick records. 
# 24 of those are also in demos.csv. Removing those.

matches_match_ids = matches['MatchId'].unique()
map_pick_match_ids = map_picks['MatchId'].unique()
demos_match_ids = demos['MatchId'].unique()
no_map_picks = []

for m in matches_match_ids:
    if m not in map_pick_match_ids:
        no_map_picks.append(m)

for m in demos_match_ids:
    if m not in map_pick_match_ids:
        no_map_picks.append(m)

no_map_picks = set(no_map_picks)

demos = remove_records(demos,['MatchId'],no_map_picks)
player_demos = remove_records(player_demos,['MatchId'],no_map_picks)
matches = remove_records(matches,['MatchId'],no_map_picks)

### Remove Weird other maps
# Three maps only appear about 5 times, removing them. 
# Note: I guess they got cleared out in a previous filter.

weird_maps = ['de_tuscan','de_cobblestone','de_cache']

# Label each record with T/F if a team involved has fewer than min_games
demos = remove_records(demos,['MapName'],weird_maps)
map_picks = remove_records(map_picks,['MapName'],weird_maps)
player_demos = remove_records(player_demos,['MapName'],weird_maps)

### Remove matches where DecisionOrder and Decision don't align
# DecisionOrder 1,2,5,6 should be 'Remove' and DecisionOrder 3,4,7 should be 'Pick'
# Picks in 1,2,5,6
wrong_decision_match_ids = list(map_picks[(map_picks['Decision']=='Pick') & ((map_picks['DecisionOrder'] == 1)|
                                                                             (map_picks['DecisionOrder'] == 2)|
                                                                             (map_picks['DecisionOrder'] == 5)|
                                                                             (map_picks['DecisionOrder'] == 6)
                                                                            )]['MatchId'])
# Removes in 3,4,7
wrong_decision_match_ids.extend(list(map_picks[(map_picks['Decision']=='Remove') & ((map_picks['DecisionOrder'] == 3)|
                                                                                    (map_picks['DecisionOrder'] == 4)|
                                                                                    (map_picks['DecisionOrder'] == 7)
                                                                                   )]['MatchId']))

demos = remove_records(demos,['MatchId'],wrong_decision_match_ids)
player_demos = remove_records(player_demos,['MatchId'],wrong_decision_match_ids)
matches = remove_records(matches,['MatchId'],wrong_decision_match_ids)
map_picks = remove_records(map_picks,['MatchId'],wrong_decision_match_ids)

### Matches with !=3 games
# Number of picks by match ID
pick_count = map_picks[map_picks['Decision']=='Pick'].groupby('MatchId').count()['MapName']
# Match Ids where num_picks does not equal 3
not_3_picks = list(pick_count[pick_count!=3].index)

demos = remove_records(demos,['MatchId'],not_3_picks)
player_demos = remove_records(player_demos,['MatchId'],not_3_picks)
matches = remove_records(matches,['MatchId'],not_3_picks)
map_picks = remove_records(map_picks,['MatchId'],not_3_picks)

### Teams with fewer than min_games
# After the other filters.

# Team Ids for teams with fewer than 25 games
min_games = 25

# Repeat to remove teams that fall below 25 games when others are removed
for i in range(8):
    print("Round ",i)
    team_record = build_team_record_df(teams,demos)
    few_games_team_ids = team_record[team_record['TotalGames']<min_games]['TeamId']

    # Label each record with T/F if a team involved has fewer than min_games
    teams = remove_records(teams,['TeamId'],few_games_team_ids)
    demos = remove_records(demos,['WinnerId','LoserId'],few_games_team_ids)
    matches = remove_records(matches,['WinnerId','LoserId'],few_games_team_ids)
    map_picks = remove_records(map_picks,['DecisionTeamId','OtherTeamId'],few_games_team_ids)
    player_demos = remove_records(player_demos,['TeamId'],few_games_team_ids)

print(teams.shape)
print(map_picks.shape)
print(matches.shape)
print(demos.shape)
print(player_demos.shape)

### Remove Updated and Created
demos = demos.drop(['Created','Updated'],axis=1)
map_picks = map_picks.drop(['Created','Updated'],axis=1)
matches = matches.drop(['Created','Updated'],axis=1)
teams = teams.drop(['Created','Updated'],axis=1)
players = players.drop(['Created','Updated'],axis=1)
player_demos = player_demos.drop(['Created','Updated'],axis=1)

### Save clean csv's
out_dir = '../data/clean/'

if not os.path.exists(data_dir):
    os.mkdir(data_dir)

demos.to_csv(out_dir+'demos.csv',index=False)
map_picks.to_csv(out_dir+'map_picks.csv',index=False)
matches.to_csv(out_dir+'matches.csv',index=False)
teams.to_csv(out_dir+'teams.csv',index=False)
players.to_csv(out_dir+'players.csv',index=False)
player_demos.to_csv(out_dir+'player_demos.csv',index=False)

print("Done")








