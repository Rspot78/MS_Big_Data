import isodate
from SPARQLWrapper import SPARQLWrapper, JSON, N3, TURTLE
from rdflib import Graph, URIRef, Literal
import json

class RdfDev:
    def __init__(self):
        """__init__
        Initialise la session de collecte
        :return: Object of class Collecte
        """
        # NE PAS MODIFIER
        self.basename = "rdfsparql.step"

    def rdfdev(self):
        """collectes
        Plusieurs étapes de collectes. VOTRE CODE VA VENIR CI-DESSOUS
        COMPLETER les méthodes stepX.
        """
        self.step1()
        self.step2()
        self.step3()
        self.step4()
        self.step5()

    def step1(self):
        stepfilename = self.basename+"1"
        result = { "typelist": []}
        # votre code ici
        with open(stepfilename, "w", encoding="utf-8") as resfile:
            json.dump(result, resfile)


    def step2(self):
        stepfilename = self.basename+"2"
        result = {}
        # votre code ici
        with open(stepfilename, "w", encoding="utf-8") as resfile:
            json.dump(result, resfile)

    def step3(self):
        stepfilename = self.basename+"3"
        result = {}
        # votre code ici
        with open(stepfilename, "w", encoding="utf-8") as resfile:
            json.dump(result, resfile)

    def step4(self):
        stepfilename = self.basename+"4"
        result = {}
        # votre code ici
        with open(stepfilename, "w", encoding="utf-8") as resfile:
            json.dump(result, resfile)

    def step5(self):
        stepfilename = self.basename+"5"
        result = {}
        # votre code ici
        with open(stepfilename, "w", encoding="utf-8") as resfile:
            json.dump(result, resfile)



if __name__ == "__main__":
    testeur = RdfDev()
    testeur.rdfdev()
