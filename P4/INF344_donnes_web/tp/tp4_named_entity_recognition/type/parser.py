'''Parses a Wikipedia file, returns page objects'''
from page import Page
__author__ = "Jonathan Lajus"

class Parser:
    def __init__(self, wikipediaFile):
        self.file = wikipediaFile
    def __iter__(self):
        title, content = None,""
        with open(self.file, encoding='utf-8') as f:
            for line in f:                
                line = line.strip()
                if not line and title is not None:
                    yield Page(title, content.rstrip())
                    title, content = None,""
                elif title is None:
                    title = line
                elif title is not None:
                    content += line + " "