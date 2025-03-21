import time
import pandas as pd
import numpy as np
import pickle
import datetime
from playerobj import Player
from pathlib import Path


class Pipeline():
    def __init__(self, data, model):
        self.data = data
        self.model = model