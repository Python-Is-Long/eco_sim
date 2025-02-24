import pickle
import os
import random
from typing import List, Dict

import matplotlib.pyplot as plt
import numpy as np
from narwhals import median
from tqdm import tqdm

from utils.simulationObjects import Config, Individual, Company, Product, ProductGroup
from utils import calculation
from sim import EconomyStats, Economy

PICKLE_FILE_PATH = "economy_simulation.pkl"
with open(PICKLE_FILE_PATH, 'rb') as f:
    x = pickle.load(f)
    print(x)