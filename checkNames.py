import pickle
from sim import *

with open('economy_simulation.pkl', 'rb') as f:
    eco = pickle.load(f)
    l = [c.name for c in eco.companies]
    print(len(l))
    print(len(set(l)))