import os
import pandas as pd
import datetime
from database import return_current_player_list, get_player_box_scores, split_box_data
from InquirerPy import inquirer
from sklearn import tree



# search to find player (Inquirer)

player = inquirer.fuzzy(
    message="Select current player",
    choices=return_current_player_list,
    match_exact=True,
).execute()

# update database

# get_player_box_scores("gilgesh01", 2024,"./2023_2024_shai_regular_season_box_scores.csv")

today = datetime.date.today()
cur_year = today.year

player_box_file = f"./{cur_year}_{player.name_as_file()}_reg_season_box_scores.csv"
#processed_file_2024_2025_shai = "./processed_2024_2025_shai_regular_season_box_scores.csv"
get_player_box_scores(player.slug, cur_year, player_box_file)
#split_box_data(file_2024_2025_shai, processed_file_2024_2025_shai)

# train on previous seasons and test on current


# decision tree for feature selection
tree_clf = tree.DecisionTreeRegressor()

