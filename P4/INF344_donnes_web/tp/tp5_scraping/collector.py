# -*- coding: utf-8 -*-
# écrit par Jean-Claude Moissinac, structure du code par Julien Romero

from sys import argv
import sys
if (sys.version_info > (3, 0)):
    from urllib.request import urlopen
    from urllib.parse import urlencode
else:
    from urllib2 import urlopen
    from urllib import urlencode
from bs4 import BeautifulSoup
from time import sleep
from re import compile

class Collecte:
    """pour pratiquer plusieurs méthodes de collecte de données"""

    def __init__(self):
        """__init__
        Initialise la session de collecte
        :return: Object of class Collecte
        """
        # DO NOT MODIFY
        self.basename = "collecte.step"
        self.name = "dupont"
        
    def collectes(self):
        """collectes
        Plusieurs étapes de collectes. VOTRE CODE VA VENIR CI-DESSOUS
        COMPLETER les méthodes stepX.
        """
        self.step0()
        self.step1()
        self.step2()
        self.step3()
        self.step4()
        self.step5()
        self.step6()

    def step0(self):
        # cette étape ne sert qu'à valider que tout est en ordre; rien à coder
        stepfilename = self.basename+"0"
        print("Comment :=>> Validation de la configuration")
        with open(stepfilename, "w", encoding="utf-8") as resfile:
            resfile.write(self.name)
        
    def step1(self):
        stepfilename = self.basename+"1"
        # get content of page www.freepatentsonline.com
        result = urlopen("http://www.freepatentsonline.com/result.html?sort=relevance&srch=top&query_txt=video&submit=&patents=on") \
            .read().decode("utf8")
        # votre code ici
        with open(stepfilename, "w", encoding="utf-8") as resfile:
            resfile.write(result)
        
    def step2(self):
        stepfilename = self.basename+"2"
        # get (again) content of page www.freepatentsonline.com, saved as a string
        html_content = urlopen("http://www.freepatentsonline.com/result.html?sort=relevance&srch=top&query_txt=video&submit=&patents=on").read()
        # get soup
        soup = BeautifulSoup(html_content, 'html.parser')
        # get links
        links = [str(link.get('href')) for link in soup.find_all('a')]
        # onvert to string
        result = "\n".join(links)
        with open(stepfilename, "w", encoding="utf-8") as resfile:
            resfile.write(result)
        
    def linksfilter(self, links):
        flinks = []
        links_to_eliminate = ["/", "/services.html", "/contact.html", "/privacy.html", "/register.html",\
                              "/tools-resources.html", "https://twitter.com/FPOCommunity", \
                                  "http://www.linkedin.com/groups/FPO-Community-4524797", \
                                      "http://www.sumobrainsolutions.com/", "None"]
        for link in links:
            valid = True
            if link in links_to_eliminate:
                valid = False
            if link.startswith('result.html') or link.startswith('http://research') or link.startswith('/search.html'):
                valid = False
            if link in flinks:
                valid = False
            if valid:
                flinks.append(link)
        return flinks
        
    def step3(self):
        stepfilename = self.basename+"3"
        # recover links from step 2 as a single string
        link_string = open('collecte.step2','r').read()
        # convert to list to accomodate linksfilter input format
        link_list = link_string.splitlines()
        # filter
        unsorted_link_list = self.linksfilter(link_list)
        # sort in alphabetical order
        sorted_link_list = sorted(unsorted_link_list)
        # convert back to string
        result = "\n".join(sorted_link_list)
        with open(stepfilename, "w", encoding="utf-8") as resfile:
            resfile.write(result)
        
    def step4(self):
        stepfilename = self.basename+"4"
        # recover links from step 3 as a single string
        link_string = open('collecte.step3','r').read()
        # convert to list and keep only 10 entries
        link_list = link_string.splitlines()[:10]
        # initiate list of raw links and common url
        raw_list = []
        url = "http://www.freepatentsonline.com/"
        # loop over the 10 links
        for link in link_list:
            # open url, parse and get hyperlinks
            html_content = urlopen(url+link).read()
            soup = BeautifulSoup(html_content, 'html.parser')
            links = [str(link.get('href')) for link in soup.find_all('a')]
            # append to raw list
            raw_list.extend(links)
            # wait for 2 seconds to avoid being blocked by servor
            sleep(1)
        # eliminate doublons (inspired from https://w3schoolsrus.github.io/python/python_howto_remove_duplicates.html)
        list_no_doublon = list(dict.fromkeys(raw_list))
        # sort in alphabetical order
        sorted_link_list = sorted(list_no_doublon)        
        # convert back to string
        result = "\n".join(sorted_link_list)  
        with open(stepfilename, "w", encoding="utf-8") as resfile:
            resfile.write(result)
        
    def contentfilter(self, link):
        # open url and parse
        html_content = urlopen(link).read()
        soup = BeautifulSoup(html_content, 'html.parser')
        # collect Inventors, Titles and Application Number
        inventors = soup.find_all(text=compile('Inventors:')) is not None
        titles = soup.find_all(text=compile('Title:')) is not None
        applications = soup.find_all(text=compile('Application Number:')) is not None
        # return True if all three words are met
        if inventors and titles and applications:
            return True
        return False

    def step5(self):
        stepfilename = self.basename+"5"
        # recover links from step 4 as a single string
        link_string = open('collecte.step4','r').read()
        # convert to list
        link_list = link_string.splitlines()
        # initiate list of interesting links, counters and urls
        interesting_links = []
        interesting_counter = 0
        link_counter = 0
        url = "http://www.freepatentsonline.com/"
        # loop over links until 10 interesting links obtain
        while interesting_counter < 10:
            link = link_list[link_counter]
            link_counter += 1
            if link.endswith(".html"):
                webpage = url + link
                if self.contentfilter(webpage):
                    interesting_links.append(link)
                    interesting_counter += 1
        # convert back to string
        result = "\n".join(interesting_links)   
        with open(stepfilename, "w", encoding="utf-8") as resfile:
            resfile.write(result)
        
    def step6(self):
        stepfilename = self.basename+"6"
        # recover links from step 5 as a single string
        link_string = open('collecte.step5','r').read()
        # convert to list and keep only the first five entries
        link_list = link_string.splitlines()[:5]
        # initiate list of inventors
        inventor_list = []
        # loop over links
        url = "http://www.freepatentsonline.com/"
        for link in link_list:
            webpage = url + link
            # open url and parse
            html_content = urlopen(webpage).read()
            soup = BeautifulSoup(html_content, 'html.parser')
            # get inventors
            inventors = soup.find_all(text=compile('Inventors:'))
            divs = [inventor.parent.parent for inventor in inventors]
            for d in divs[0].descendants:
                if d.name == 'div' and d.get('class', '') == ['disp_elm_text']:
                    inventorlist = d.text
                    inventor_list.append(inventorlist)
        # convert back to string
        result = "\n".join(inventor_list)
        with open(stepfilename, "w", encoding="utf-8") as resfile:
            resfile.write(result)
        
if __name__ == "__main__":
    collecte = Collecte()
    collecte.collectes()
