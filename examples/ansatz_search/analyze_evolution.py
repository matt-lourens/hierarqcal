# %%
import os
import re
from collections import namedtuple
import numpy as np
import sympy as sp
import matplotlib.pyplot as plt
from hierarqcal import (
    Qhierarchy,
    Qcycle,
    Qpermute,
    Qmask,
    Qunmask,
    Qpivot,
    Qinit,
    Qmotif,
    Qmotifs,
    plot_motif,
    plot_circuit,
    Qunitary,
)
import dill
from utils import *
 
# list experiments dir, get highest id and increment
EXP_ID = 0 if False == True else max([int(x) for x in os.listdir(f"{PATH}/experiments")])

# find max generation
regex = r"memory_table_(\d+).pkl"
files = os.listdir(os.path.join(PATH, "experiments", f"{EXP_ID}"))
generation = max([int(re.search(regex, file_name).group(1)) for file_name in files if re.search(regex, file_name)])
generation = 6
with open(f"{PATH}/experiments/{EXP_ID}/memory_table_{generation}.pkl", "rb") as file:
    memory_table = dill.load(file)
# sort by fitness
ind=0
memory_table = sorted(memory_table, key=lambda x: x.fitness)
best_motif = Qinit(memory_table[ind].nq[-1]) + memory_table[ind].motif
# plot_circuit(best_motif[1].mapping.hierq)
plot_circuit(best_motif)
print(memory_table[ind].fitness)
print(memory_table[ind].energy)
