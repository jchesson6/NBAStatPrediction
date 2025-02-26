import os
import pandas as pd
import pickle
import datetime
from playerobj import Player
from basketball_reference_web_scraper import client
from basketball_reference_web_scraper.data import OutputType, OutputWriteOption
from sklearn.preprocessing import MinMaxScaler


def update_database(season_end_year=None):

    if season_end_year == None:
        season_end_year = datetime.datetime.now().year

    #create a file name using season
    #os.remove("2024_2025_player_season_totals.csv")
    season_start_year = season_end_year - 1
    total_csv_file = f"./{season_start_year}_{season_end_year}_player_season_totals.csv"
    schedule_csv_file = f"./{season_start_year}_{season_end_year}_season_schedule.csv"
    player_database_file = f"{season_start_year}_{season_end_year}_player_database.dat"

    # pull season totals
    client.players_season_totals(
        season_end_year=season_end_year,
        output_type=OutputType.CSV, 
        output_file_path=total_csv_file
    )

    client.season_schedule(
    season_end_year=season_end_year, 
    output_type=OutputType.CSV, 
    output_file_path=schedule_csv_file
    )

    #process the schedule file - add GAME PLAYED, CHANGE DATES, 

    #create player database with slug, Name - Team - Position
    # will need to handle duplicated players on basketball reference that have switched teams somehow
    totals_df = pd.read_csv(total_csv_file)
    player_list = []

    for row in totals_df.itertuples():
        player = Player(row.slug, row.name, row.team, row.positions)
        player_list.append(player)
        # will probably implement pulling the box scores for each player and combining them into a single dataframe and csv
        # could possibly add the season schedule as a player object property
    
    # save database as a pickle file
    with open(player_database_file, 'wb') as player_database:
        pickle.dump(player_list, player_database)

    return player_list
   

def return_current_player_list(season_end_year=None):

    if season_end_year == None:
        season_end_year = datetime.datetime.now().year

    season_start_year = season_end_year - 1
    player_database_file = f"{season_start_year}_{season_end_year}_player_database.dat"

    with open(player_database_file, 'rb') as player_database:
        player_list = pickle.load(player_database)

    return player_list



def get_player_box_scores(player, season_end_year=None, process=True):

    #create file name using player name and season if storing all csvs
    if season_end_year == None:
        season_end_year = datetime.datetime.now().year

    season_start_year = season_end_year - 1
    player_box_file = f"./{season_start_year}_{season_end_year}_{player.name_as_file()}_reg_season_box_scores.csv"

    client.regular_season_player_box_scores(
        player_identifier=player.slug, 
        season_end_year=season_end_year, 
        output_type=OutputType.CSV, 
        output_file_path=player_box_file
    )
    #this will probably move into the update database and be replaced with reading a large csv

    #create to date ppg, rpg, apg, etc stats
    if process:
        process_box_scores(player_box_file)

    #potentially create a data quality report (ECE5464 slide 5)

    #add to the overall csv containing all player box scores


def process_box_scores(filename):

    orig_df = pd.read_csv(filepath_or_buffer=filename)

    # may need to remove inactive games but currently seems like they are already removed

    # convert seconds to minutes
    minutes = orig_df['seconds_played'].div(60.0)
    orig_df.insert(11, 'minutes_played', minutes)

    # add rebounds together to get full rebounds per game
    rebounds = orig_df['offensive_rebounds'] + orig_df['defensive_rebounds']
    orig_df.insert(17, 'total_rebounds', rebounds)

    # perform normalization and data preparation

    # - 1-hot encode team and probably opponent

    # - binary encode active status and outcome


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



def normalize_and_split_box(player, season_end_year=None, target="points"):

    if season_end_year == None:
        season_end_year = datetime.datetime.now().year

    season_start_year = season_end_year - 1
    player_box_file = f"./{season_start_year}_{season_end_year}_{player.name_as_file()}_reg_season_box_scores.csv"

    df = pd.read_csv(player_box_file)

    scaler = MinMaxScaler()
    
    # remove unnecessary column
    # - dont know if fantasy points are useful right now
    # removing all string columns to just test on stats first. Opponent information and extra factors will be added later.
    #x = df.drop(columns=['active', 'plus_minus', 'team', 'location', 'opponent', 'outcome', 'game_score', 'fpts'])


    # split into inputs and outputs
    # output for a multioutput model (NOT USING RIGHT NOW)
    # y = df[['points_scored', 'total_rebounds', 'assists', 'steals', 'blocks', 'turnovers', 'made_field_goals',
           # 'attempted_field_goals', 'made_three_point_field_goals']]

    # customize data based on desired output
    if target == "total_rebounds":
        y = df['total_rebounds']
    elif target == "assists":
        y = df['assists']
    else:
        df = df[['seconds_played', 'minutes_played','made_field_goals', 'attempted_field_goals', 'made_three_point_field_goals', 
                'attempted_three_point_field_goals', 'made_free_throws', 'attempted_free_throws', 'ppg_td', 'points_scored']]
        
        x = df.drop(columns=['points_scored'])
        x_labels = x.columns.to_list()
        y = df[['points_scored']]
        y_labels = y.columns.to_list()

        x_t = scaler.fit_transform(x,y)
        y_t = scaler.fit_transform(y)

    return x_t, y_t, x_labels, y_labels