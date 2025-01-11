import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import dash_bootstrap_components as dbc
import pandas as pd
import pickle
import os
from datetime import datetime
import threading
import time
from sim import EconomyStats



with open('simulation_stats.pkl', 'rb') as f:
    data = pickle.load(f)