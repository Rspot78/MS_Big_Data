#!/usr/bin/python3
# -*- coding: utf-8 -*-

import getpage
import unittest
import time

# http://stackoverflow.com/a/2648359
class MyTestCase(unittest.TestCase):
    def setUp(self):
        pass

    def tearDown(self):
        time.sleep(1)

    def containsThatContains(self, l, s):
        for a in l:
            if s.lower() in a.lower():
                return True
        return False


class MyTests(MyTestCase):
    def testQ2title(self):
        article = "Bonjour"
        title, _ = getpage.getRawPage(article)
        self.assertEqual(title, "Bonjour")

    def testQ2contents(self):
        article = "Bonjour"
        _, contents = getpage.getRawPage(article)
        self.assertTrue(contents.startswith("<div"))

    def testQ3title(self):
        article = "Bonjour"
        title, _ = getpage.getPage(article)
        self.assertEqual(title, "Bonjour")

    def testQ3content1(self):
        article = "Bonjour"
        link = "Salutation"
        _, links = getpage.getPage(article)
        self.assertTrue(self.containsThatContains(links, link))

    def testQ3content2(self):
        article = "Bonjour"
        link = "Correspondance"
        _, links = getpage.getPage(article)
        self.assertTrue(self.containsThatContains(links, link))

    def testQ3content3(self):
        article = "Seine"
        link = "Troyes"
        _, links = getpage.getPage(article)
        self.assertTrue(self.containsThatContains(links, link))

    def testFina3(self):
        article = "Fondo Strategico Italiano"
        links_gold = ['Fonds de placement', 'Fonds souverain', 'Bulgari',
                'LVMH', 'Parmalat', 'Lactalis',
                "Fonds stratégique d'investissement",
                'Cassa depositi e prestiti', 'Fintecna',
                'Avio']
        title, links = getpage.getPage(article)
        self.assertEqual(links, links_gold)



if __name__ == "__main__":
    # https://stackoverflow.com/a/2713010/414272
    log_file = 'test_output.txt'
    f = open(log_file, "w")
    runner = unittest.TextTestRunner(f)
    suite = unittest.TestLoader().loadTestsFromTestCase(MyTests)
    ret = runner.run(suite)
    total = ret.testsRun
    bad = len(ret.failures) + len(ret.errors)
    print("Comment :=>> %d tests failed out of %d total tests" % (bad, total))
    print("Grade :=>> %d" % int(20.*(total-bad)/total))
    f.close()

