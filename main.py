import os
import pandas as pd
import inquirer
from database import update_player_database, get_player_box_scores, split_box_data
from sklearn import tree



# search to find player (Inquirer)


# update database

update_player_database(2025)
# get_player_box_scores("gilgesh01", 2024,"./2023_2024_shai_regular_season_box_scores.csv")

file_2024_2025_shai = "./2024_2025_shai_regular_season_box_scores.csv"
processed_file_2024_2025_shai = "./processed_2024_2025_shai_regular_season_box_scores.csv"
get_player_box_scores("gilgesh01", 2025, file_2024_2025_shai)
split_box_data(file_2024_2025_shai, processed_file_2024_2025_shai)

# train on previous seasons and test on current


# decision tree for feature selection
tree_clf = tree.DecisionTreeRegressor()

