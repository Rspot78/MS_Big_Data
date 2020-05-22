#!/usr/bin/env python3

import sqlite3
import re
from math import log
from shared import extractText, stem
from collections import defaultdict
conn = sqlite3.connect('data.db')
cursor = conn.cursor()

query = input()
queryWords = [stem(w) for w in query.split()]


# compute best query solution and output them
