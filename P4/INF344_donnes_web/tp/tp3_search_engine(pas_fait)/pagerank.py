import sqlite3
import re
from math import log
from shared import extractText, neighbors
from collections import defaultdict


NB_ITERATIONS = 50
ALPHA = 0.15

conn = sqlite3.connect('data.db')
cursor = conn.cursor()

#compute and store pagerank

conn.commit()

    
