import pandas as pd
import sqlite3

conn = sqlite3.connect('database.sqlite')
teams_data = pd.read_sql_query("SELECT * from team", conn)
match_data = pd.read_sql_query("SELECT * from match", conn)
conn.close()


def fetchTeamNameByID(id):
    matching_row = teams_data[teams_data['team_api_id'] == id]
    team_name = matching_row['team_long_name'].values[0]
    return team_name


# Create a new column 'home_team_win' and populate it with 1 or 0
# match_data['home_team_win'] = match_data.apply(lambda row: 1 if row['home_team_goal'] > row['away_team_goal'] else 0, axis=1)
match_data['home_team_win'] = match_data.apply(
    lambda row: 1 if row['home_team_goal'] > row['away_team_goal'] else 0 if row['home_team_goal'] == row['away_team_goal'] else -1, axis=1)

# get home and away team name
match_data['home_team_name'] = match_data.apply(
    lambda row: fetchTeamNameByID(row['home_team_api_id']), axis=1)
match_data['away_team_name'] = match_data.apply(
    lambda row: fetchTeamNameByID(row['away_team_api_id']), axis=1)

# match_data.to_csv('match_data.csv')

# take only features that are relevant
features = ['date', 'home_team_api_id', 'away_team_api_id', 'home_team_name', 'away_team_name',
            'home_team_win', 'possession', 'shoton', 'shotoff', 'B365H', 'B365D', 'B365A', 'BWH', 'BWA']
match_data.dropna(inplace=True)
match_data_with_features_only = match_data[features]

# write to csv
match_data_with_features_only.to_csv('match_data_with_features_only.csv')
