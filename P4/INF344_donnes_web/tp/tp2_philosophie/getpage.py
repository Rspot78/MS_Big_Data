#!/usr/bin/python3
# -*- coding: utf-8 -*-

from bs4 import BeautifulSoup
from json import loads
from urllib.request import urlopen
from urllib.parse import urlencode
import ssl
from urllib.parse import unquote

# initiate the cache variable (as a dictionary)
cache = {}

def getJSON(page):
    params = urlencode({
      'format': 'json',  # TODO: compléter ceci
      'action': 'parse',  # TODO: compléter ceci
      'prop': 'text',  # TODO: compléter ceci
      'redirects': 'true',
      'page': page})
    API = "https://fr.wikipedia.org/w/api.php"  # TODO: changer ceci
    # désactivation de la vérification SSL pour contourner un problème sur le
    # serveur d'évaluation -- ne pas modifier
    gcontext = ssl.SSLContext()
    response = urlopen(API + "?" + params, context=gcontext)
    return response.read().decode('utf-8')


def getRawPage(page):
    parsed = loads(getJSON(page))
    try:
        title = parsed['parse']['title']  # TODO: remplacer ceci
        content = parsed['parse']['text']['*']  # TODO: remplacer ceci
        return title, content
    except KeyError:
        # La page demandée n'existe pas
        return None, None


def getPage(page):
    # if page is in cache, update from cache without calling API
    if page in cache:
        return page, cache[page]
    else:
        title, content = getRawPage(page)
        if title is None:
            return None, []
        else:
            # parse page content
            soup = BeautifulSoup(content, 'html.parser')
            # find list of <p> elements that are direct children of <div> elements
            filtered_list = soup.div.find_all("p", recursive=False)
            href_list = []
            # find all link tags in filtered_list and append hyperlinks to href_list
            for entry in filtered_list:
                for link_tag in entry.find_all('a'):
                    link = link_tag.get('href')
                    # eliminate link if it is None type
                    if link:
                        # eliminate links that do not start with "/wiki/" or contain "redlink"
                        if link.startswith("/wiki/") and "redlink" not in link:
                            # eliminate the first 6 caracters which correspond to the prefix "/wiki/"
                            shortened_link = link[6:]
                            # unquote to read correctly non-ascii characters
                            unquoted_link = unquote(shortened_link)
                            # remove any part of the link following a hashtag
                            hashtag_link = unquoted_link.split("#")[0]
                            # replace underscores with spaces
                            underscore_link = hashtag_link.replace('_', ' ')
                            # if the link contains ':' or has become empty, ignore it
                            if (':' in  underscore_link) or (not underscore_link):
                                pass
                            # else, concatenate link to the list
                            else:
                                href_list.append(underscore_link)
            # remove duplicate from list (note: method taken from https://www.w3schools.com/python/python_howto_remove_duplicates.asp)
            href_list = list(dict.fromkeys(href_list))
            # keep at most 10 entries
            element_number = min(10, len(href_list))
            trimmed_href_list = href_list[:element_number]
            # update cache
            cache[title] = trimmed_href_list
            return title, trimmed_href_list


if __name__ == '__main__':
    # Ce code est exécuté lorsque l'on exécute le fichier
    print("Ça fonctionne !")
    
    # Voici des idées pour tester vos fonctions :
    # print(getJSON("Utilisateur:A3nm/INF344"))
    # print(getRawPage("Utilisateur:A3nm/INF344"))
    # print(getRawPage("Histoire"))

