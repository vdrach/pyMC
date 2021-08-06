import numpy as np
import core
import random

import matplotlib.pyplot as plt

a = core.Lattice((3,4,5),BC='periodic')
a.print_dict_ix_pos()     
a.get_pos(3)

a.get_neighbours(59)

