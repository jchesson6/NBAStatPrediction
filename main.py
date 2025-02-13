import os
import pandas as pd
from database import get_season_totals, get_player_box_scores
from sklearn import tree


# update database
get_season_totals(2025)
get_player_box_scores("gilgesh01", 2025)

#search to find player (Inquirer)



# decision tree for feature selection
tree_clf = tree.DecisionTreeClassifier()