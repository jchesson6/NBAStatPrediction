import os
import datetime, time
from database import update_database, return_current_player_list, get_player_box_scores, normalize_and_split_box
from InquirerPy import inquirer
from sklearn import tree, pipeline
from sklearn.multioutput import MultiOutputRegressor
from sklearn.linear_model import Ridge



# update database
#today = datetime.date.today()
#cur_year = today.year

#set an if condition or automate this in a new script
update_database()
player_list = return_current_player_list()

# search to find player (Inquirer)
player = inquirer.fuzzy(
    message="Select current player",
    choices=player_list,
    match_exact=True,
).execute()

# select which outputs to predict

# pick a game from the schedule to estimate for


#player_box_file = f"./{cur_year}_{player.name_as_file()}_reg_season_box_scores.csv"
#processed_file_2024_2025_shai = "./processed_2024_2025_shai_regular_season_box_scores.csv"

#get player box score
get_player_box_scores(player)
X, y, feature_labels, target_label = normalize_and_split_box(player)
print(X[:3])
print(y[:3])

# train on previous seasons and test on current


# will probably go with multiple models for each predicted stat


