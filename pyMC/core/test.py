import numpy as np
import geometry


a = lattice.Lattice((3,4,5),BC='periodic')
a.print_dict_ix_pos()     
a.get_pos(3)
