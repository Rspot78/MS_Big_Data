# clear
from IPython import get_ipython
get_ipython().magic('reset -sf')
%clear

# dashboard module
from dnn_viewer import *

# path to folder
path = "/home/romain/Bureau/fil_rouge"

# call dashboard
dnn_viewer(path)