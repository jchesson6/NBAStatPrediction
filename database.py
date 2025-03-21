import time
import pandas as pd
import numpy as np
import pickle
import datetime
from playerobj import Player
from basketball_reference_web_scraper import client
from basketball_reference_web_scraper.data import OutputType, Position, Team, Location, Outcome
from sklearn.preprocessing import MinMaxScaler
from pathlib import Path


def update_database(season_end_year=None):

    if season_end_year == None:
        season_end_year = datetime.datetime.now().year

    Path("data").mkdir(parents=True, exist_ok=True)
    

    #create file names using season
    season_start_year = season_end_year - 1
    total_csv_file = f"./data/{season_start_year}_{season_end_year}_player_season_totals.csv"
    schedule_csv_file = f"./data/{season_start_year}_{season_end_year}_season_schedule.csv"
    standings_csv_file = f"./data/{season_start_year}_{season_end_year}_season_standings.csv"
    player_database_file = f"/data/{season_start_year}_{season_end_year}_player_database.dat"
    raw_player_box_file = f"data/{season_start_year}_{season_end_year}_raw_player_box_scores.csv"
    player_box_file = f"data/{season_start_year}_{season_end_year}_player_box_scores.csv"

    # pull season totals
    client.players_season_totals(
        season_end_year=season_end_year,
        output_type=OutputType.CSV, 
        output_file_path=total_csv_file
    )

    # pull season schedule
    client.season_schedule(
    season_end_year=season_end_year, 
    output_type=OutputType.CSV, 
    output_file_path=schedule_csv_file
    )

    #process the schedule file - add GAME PLAYED, CHANGE DATES, 


    # pull team standings
    client.standings(
    season_end_year=season_end_year,
    output_type=OutputType.CSV, 
    output_file_path=standings_csv_file
    )


    #create player database with slug, Name - Team - Position
    # will need to handle duplicated players on basketball reference that have switched teams somehow
    totals_df = pd.read_csv(total_csv_file)
    player_list = []
    all_player_dfs = []

    for row in totals_df.itertuples():
        player = Player(row.slug, row.name, row.team, row.positions)
        player_list.append(player)
        # could possibly add the season schedule as a player object property

        # pull the box scores for each player as a list of dictionaries
        box = client.regular_season_player_box_scores(
        player_identifier=row.slug, 
        season_end_year=season_end_year
        )  

        # convert to dataframe and add player identification information before adding to list of dataframes
        cur_df = pd.DataFrame(box)
        cur_df.insert(0, 'position', row.positions)
        cur_df.insert(0, 'name', row.name)
        cur_df.insert(0, 'slug', row.slug)
        all_player_dfs.append(cur_df)

        # wait so the request limit is not hit
        time.sleep(4)
    
    # combine all the dataframes in the list and convert to csv
    all_player_box = pd.concat(all_player_dfs, ignore_index=True)
    all_player_box.to_csv(path_or_buf=raw_player_box_file)

    # edit dataframe to include extra columns
    edit_player_box = edit_box_scores(all_player_box)
    edit_player_box.to_csv(player_box_file)


    # save database as a pickle file
    with open(player_database_file, 'wb') as player_database:
        pickle.dump(player_list, player_database)

   

def return_current_player_list(season_end_year=None):

    if season_end_year == None:
        season_end_year = datetime.datetime.now().year

    season_start_year = season_end_year - 1
    player_database_file = f"{season_start_year}_{season_end_year}_player_database.dat"

    with open(player_database_file, 'rb') as player_database:
        player_list = pickle.load(player_database)

    return player_list



def get_player_box_scores(player, season_end_year=None, process=True):

    #CHANGE THIS FUNCTION TO RETRIVING BOX SCORE FROM THE TOTAL CSV

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
    

    #potentially create a data quality report (ECE5464 slide 5)

    #add to the overall csv containing all player box scores


def edit_box_scores(orig_df):

    # may need to remove inactive games but currently seems like they are already removed

    # convert seconds to minutes
    minutes = orig_df['seconds_played'].div(60.0)
    orig_df.insert(10, 'minutes_played', minutes)

    # add rebounds together to get full rebounds per game
    rebounds = orig_df['offensive_rebounds'] + orig_df['defensive_rebounds']
    orig_df.insert(19, 'total_rebounds', rebounds)

    # calculate fantasy points for each game
    orig_df['fpts'] = orig_df.apply(get_fantasy_points, axis=1)

    # cumulative rolling averages as to date stats
    orig_df['ppg_td'] = orig_df['points_scored'].expanding().mean().round(1)
    orig_df['rpg_td'] = orig_df['total_rebounds'].expanding().mean().round(1)
    orig_df['apg_td'] = orig_df['assists'].expanding().mean().round(1)
    orig_df['spg_td'] = orig_df['steals'].expanding().mean().round(1)
    orig_df['bpg_td'] = orig_df['blocks'].expanding().mean().round(1)
    orig_df['tpg_td'] = orig_df['turnovers'].expanding().mean().round(1)
    
    return orig_df


def get_fantasy_points(row):
    # can substitute any values for fantasy but these correspond to my leagues
    return (
        row['points_scored'] + row['total_rebounds'] + 2*row['assists'] + 3*row['steals'] 
        + 3*row['blocks'] - 2*row['turnovers'] + 2*row['made_field_goals'] 
        - row['attempted_field_goals'] + row['made_three_point_field_goals']
            )



def process_box(scalerx, scalery, playerscaler, outscaler, csvfile=None, season_end_year=None, player=None, target="points"):

    if season_end_year == None:
        season_end_year = datetime.datetime.now().year
        season_start_year = season_end_year - 1

    train_file = f"data/{season_start_year}_{season_end_year}_training_box_scores.csv"
    test_file = f"data/{season_start_year}_{season_end_year}_testing_box_scores.csv"

    if csvfile == None:
        csvfile = f"data/{season_start_year}_{season_end_year}_player_box_scores.csv"

    player_list = return_current_player_list()

    df = pd.read_csv(csvfile)


    # doing this for now to reduce amount of data
    #df.drop(columns=['active', 'plus_minus', 'team', 'location', 'opponent', 'outcome', 'game_score', 'fpts'])

    # perform normalization and data preparation

    # - 1-hot encode team and probably opponent

    # - binary encode active status and outcome
    
    # remove unnecessary column
    # - dont know if fantasy points are useful right now
    # removing all string columns to just test on stats first. Opponent information and extra factors will be added later.
    #x = df.drop(columns=['active', 'plus_minus', 'team', 'location', 'opponent', 'outcome', 'game_score', 'fpts'])


    # split data into training and testing data by exracting last n games for each player
    n = 3 # number of rows to keep for testing
    train_df = []
    test_df = []
    player_box_l = []
    
    for cur_player in player_list:
        cur_player_box = df[df['slug'] == cur_player.slug]

        train_sec = cur_player_box.iloc[:-n]
        train_df.append(train_sec)

        test_sec = cur_player_box.tail(n)
        test_df.append(test_sec)

        if cur_player.name == player.name:
            player_box_l.append(cur_player_box)

    # combine dataframaes and convert to csv
    train_data = pd.concat(train_df, ignore_index=True)
    train_data.to_csv(path_or_buf=train_file)

    test_data = pd.concat(test_df, ignore_index=True)
    test_data.to_csv(path_or_buf=test_file)

    
    player_box = pd.DataFrame(player_box_l[0])

    # split into inputs and outputs
    # customize data based on desired output
    if target == "points":

        scale_cols = ['seconds_played', 'minutes_played','made_field_goals', 'attempted_field_goals', 'made_three_point_field_goals', 
                'attempted_three_point_field_goals', 'made_free_throws', 'attempted_free_throws', 'points_scored']
        
        #df[scale_cols] = scaler.fit_transform(df[scale_cols]) #normalize only input and output columns

        #may include slug but for now only keep numeric
        x_train = train_data[['seconds_played', 'minutes_played', 'made_field_goals', 'attempted_field_goals', 'made_three_point_field_goals', 
                'attempted_three_point_field_goals', 'made_free_throws', 'attempted_free_throws', 'points_scored']]
        y_train = train_data[['points_scored']]

        x_test = test_data[['seconds_played', 'minutes_played','made_field_goals', 'attempted_field_goals', 'made_three_point_field_goals', 
                'attempted_three_point_field_goals', 'made_free_throws', 'attempted_free_throws', 'points_scored']]
        y_test = test_data[['points_scored']]

        
        player_box = player_box[['seconds_played', 'minutes_played','made_field_goals', 'attempted_field_goals', 'made_three_point_field_goals', 
            'attempted_three_point_field_goals', 'made_free_throws', 'attempted_free_throws', 'points_scored']]
        player_in = player_box.tail(n)
        player_out = player_in[['points_scored']]
        player_in_s = playerscaler.fit_transform(player_in)
        outscaler.fit(player_out)


        xtrain_s = scalerx.fit_transform(x_train)
        ytrain_s = scalery.fit_transform(y_train)
        xtest_s = scalerx.fit_transform(x_test)
        ytest_s = scalery.fit_transform(y_test)


    elif target == "rebounds":


        x = df[['seconds_played', 'minutes_played', 'offensive_rebounds', 'defensive_rebounds']]
        y = df[['total_rebounds']]


    elif target == "assists":

        x = df[['seconds_played', 'minutes_played']]
        y = df[['assists']]

    else:
        return #for options not currently implemented
    

    x_labels = x_train.columns.to_list()
    y_labels = y_train.columns.to_list()

    return xtrain_s, ytrain_s, xtest_s, ytest_s, player_in_s, x_labels, y_labels, outscaler


def split_box_data(df, target="points"):

    # customize data based on desired output
    if target == "rebounds":
        x = df[['seconds_played', 'minutes_played', 'offensive_rebounds', 'defensive_rebounds']]
        y = df[['total_rebounds']]
    elif target == "assists":
        y = df[['assists']]
    elif target == "points":
        x = df[['seconds_played', 'minutes_played','made_field_goals', 'attempted_field_goals', 'made_three_point_field_goals', 
                'attempted_three_point_field_goals', 'made_free_throws', 'attempted_free_throws', 'ppg_td']]
        y = df[['points_scored']]
    else:
        return #for options not currently implemented
        

    x_labels = x.columns.to_list()
    y_labels = y.columns.to_list()

    return x, y, x_labels, y_labels


def generate_windowed_sequence_data(x_orig, y_orig, window_size):

    x_seq = []
    y_seq = []

    for i in range(len(x_orig)-window_size):
        x = x_orig[i:(i+window_size)]
        y = y_orig[(i+1):(i+window_size+1)]
        x_seq.append(x)
        y_seq.append(y)

    return np.array(x_seq), np.array(y_seq)

def retrieve_player_input(player, window_size):
    
    season_end_year = datetime.datetime.now().year
    season_start_year = season_end_year - 1
    csvfile = f"data/{season_start_year}_{season_end_year}_player_box_scores.csv"

    x, y, features, labels = process_box(csvfile=csvfile, player=player)

    return x.tail(window_size)

    