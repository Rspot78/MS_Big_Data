#!/usr/bin/env python3

import sqlite3
import re
from math import log
from shared import extractListOfWords, stem
from collections import defaultdict
conn = sqlite3.connect('data.db')
cursor = conn.cursor()

# compute the inverted index and the idf and store them

conn.commit()
