import os
import pandas as pd
from basketball_reference_web_scraper import client
from basketball_reference_web_scraper.data import OutputType


def get_season_totals(season_end_year):

    #create a file name using season
    #os.remove("2024_2025_player_season_totals.csv")

    client.players_season_totals(
        season_end_year=2025, 
        output_type=OutputType.CSV, 
        output_file_path="./2024_2025_player_season_totals.csv"
    )

    #create player database with slug, Name - Team - Position

    #create the per game stats
    #create_per_game_stats()



def get_player_box_scores(player, season_end_year):

    #create file name using player name and season if storing all csvs

    client.regular_season_player_box_scores(
        player_identifier=player, 
        season_end_year=season_end_year, 
        output_type=OutputType.CSV, 
        output_file_path="./2024_2025_shai_regular_season_box_scores.csv"
    )

    #create current ppg, rpg, apg, etc stats

    #add to the overall csv containing all player box scores


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