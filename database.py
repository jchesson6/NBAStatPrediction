import os
import pandas as pd
import pickle
from playerobj import Player
from basketball_reference_web_scraper import client
from basketball_reference_web_scraper.data import OutputType, OutputWriteOption


def update_player_database(season_end_year):

    #create a file name using season
    #os.remove("2024_2025_player_season_totals.csv")
    season_start_year = season_end_year - 1
    csv_file = f"./{season_start_year}_{season_end_year}_player_season_totals.csv"
    database_file = f"{season_start_year}_{season_end_year}_player_database.dat"

    client.players_season_totals(
        season_end_year=season_end_year,
        output_type=OutputType.CSV, 
        output_file_path=csv_file
    )

    totals_df = pd.read_csv(csv_file)

    #create player database with slug, Name - Team - Position
    # will need to handle duplicated players on basketball reference that have switched teams somehow
    player_list = []

    for row in totals_df.itertuples():
        player = Player(row.slug, row.name, row.team, row.positions)
        player_list.append(player)
        # will probably implement pulling the box scores for each player and combining them into a single dataframe and csv
    
    # save database as a pickle file
    with open(database_file, 'wb') as database:
        pickle.dump(player_list, database)

    return player_list
   

def return_current_player_list(name):

    return update_player_database(2025)



def get_player_box_scores(slug, season_end_year, filename, process=True):

    #create file name using player name and season if storing all csvs

    client.regular_season_player_box_scores(
        player_identifier=slug, 
        season_end_year=season_end_year, 
        output_type=OutputType.CSV, 
        output_file_path=filename
    )
    #this will probably move into the update database and be replaced with reading a large csv

    #create to date ppg, rpg, apg, etc stats
    if process:
        process_box_scores(filename)

    #add to the overall csv containing all player box scores


def process_box_scores(filename):

    orig_df = pd.read_csv(filepath_or_buffer=filename)

    # may need to remove inactive games but currently seems like they are already removed

    # add rebounds together to get full rebounds per game
    rebounds = orig_df['offensive_rebounds'] + orig_df['defensive_rebounds']
    orig_df.insert(17, 'total_rebounds', rebounds)

    # calculate fantasy points for each game
    orig_df['fpts'] = orig_df.apply(get_fantasy_points, axis=1)

    # cumulative rolling averages as to date stats
    orig_df['ppg_td'] = orig_df['points_scored'].expanding().mean().round(1)
    orig_df['rpg_td'] = orig_df['total_rebounds'].expanding().mean().round(1)
    orig_df['apg_td'] = orig_df['assists'].expanding().mean().round(1)
    orig_df['spg_td'] = orig_df['steals'].expanding().mean().round(1)
    orig_df['bpg_td'] = orig_df['blocks'].expanding().mean().round(1)
    orig_df['tpg_td'] = orig_df['turnovers'].expanding().mean().round(1)
    
    orig_df.to_csv(path_or_buf=filename)


def get_fantasy_points(row):
    # can substitute any values for fantasy but these correspond to my leagues
    return (
        row['points_scored'] + row['total_rebounds'] + 2*row['assists'] + 3*row['steals'] 
        + 3*row['blocks'] - 2*row['turnovers'] + 2*row['made_field_goals'] 
        - row['attempted_field_goals'] + row['made_three_point_field_goals']
            )


def create_per_game_stats():

    total_df = pd.read_csv(filepath_or_buffer="2024_2025_player_season_totals.csv")
    total_df['rebounds'] = total_df['offensive_rebounds'] + total_df['defensive_rebounds']
    
    per_game_df = total_df[['slug', 'name', 'positions', 'age', 'team']].copy()

    per_game_df['PPG'] = total_df['points'].div(total_df['games_played'])
    per_game_df['RPG'] = total_df['rebounds'].div(total_df['games_played'])
    per_game_df['APG'] = total_df['assists'].div(total_df['games_played'])
    per_game_df['SPG'] = total_df['steals'].div(total_df['games_played'])
    per_game_df['BPG'] = total_df['blocks'].div(total_df['games_played'])
    per_game_df['TOPG'] = total_df['turnovers'].div(total_df['games_played'])

    per_game_df.to_csv(path_or_buf="2024_2025_player_season_totals_per_game.csv")


def split_box_data(filename, cleaned_filename):

    df = pd.read_csv(filename)
    
    # remove unnecessary column
    # - dont know if fantasy points are useful right now
    # removing all string columns to just test on stats first. Opponent information and extra factors will be added later.
    x = df.drop(columns=['active', 'plus_minus', 'team', 'location', 'opponent', 'outcome', 'game_score', 'fpts'])

    # convert string column to numerical

    # split into inputs and outputs
    y = df[['points_scored', 'total_rebounds', 'assists', 'steals', 'blocks', 'turnovers', 'made_field_goals',
             'attempted_field_goals', 'made_three_point_field_goals']]
    
    x.to_csv(cleaned_filename)
