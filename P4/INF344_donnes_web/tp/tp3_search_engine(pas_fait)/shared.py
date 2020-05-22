from bs4 import BeautifulSoup
import re

# Remove the part after the last '/' of an URL
# https://a.com/b/c/d becomes https://a.com/b/c
def baseURL(url):
    return "/".join(url.split("/")[:-1])

# Transform a local link (e.g. starting without protocols) into global
# links and return None when the link is not a local link
def globalLink(oldURL,newURL):
    if len(newURL)<2:
        return None
    if len(newURL)>8 and (newURL[0:8]=="https://" or newURL[0:7]=="http://" or newURL[0:6]=="ftp://"):
        return None
    if newURL[0]=='/':
        return "/".join(oldURL.split("/")[0:3])+"/"+newURL[1:]
    rootedBaseURL = baseURL(oldURL)
    while len(newURL)>=3 and newURL[0:3]=='../':
        rootedBaseURL=baseURL(rootedBaseURL)
        newURL = newURL[3:]
    while newURL[0:2]=='./':
        newURL=newURL[2:]
    return rootedBaseURL+"/"+newURL

blacklist = [
    '[document]',
    'noscript',
    'header',
    'html',
    'meta',
    'head',
    'input',
    'script',
    'style'
    ]

# Transforms an HTML page into a list of words ;
# technically it returns a generator but that can
# be tranformed into a list with list(extractListOfWords(content))
def extractListOfWords(content):
    soup = BeautifulSoup(content, 'html.parser')
    textTags = soup.find_all(text=True)    # find all tags with texts
    # this regex tries to find words that should not contain any of these weird chars.
    regex=re.compile(r"[^ \n'’)(\]\[ {}\\]+")
    for t in textTags:
        if t.parent.name not in blacklist: # we don't want to index thoses texts
            for e in regex.findall(t):
                yield e

# Transforms an HTML page into its textual content
def extractText(content):
    soup = BeautifulSoup(content, 'html.parser')
    textTags = soup.find_all(text=True)    # find all tags with texts
    # this regex tries to find words that should not contain any of these weird chars.
    text = ""
    for t in textTags:
        if t.parent.name not in blacklist: # we don't want to index thoses texts
            text+=" "+t
    return test


# Very simple stemming function that lower word and removes final s and e.
def stem(word):
    word = word.lower()
    if len(word)>0 and word[-1] == 's':
        word = word[0:-1]
    if len(word)>0 and word[-1] == 'e':
        word = word[0:-1]
    return word

# Returns the list of links on a webpage.
# The links returned are only the local links but
# expanded (i.e. we add the https://wiki.jachiet.com)
#
# BEWARE the links returned by this function are the links that our
# algorithm will query but not the links that are answered from the
# server. To get the corresponding respURL you need to look up in the
# database
def neighbors(pageContent, respURL):
    m = re.findall('href="([^"#]*)', pageContent)
    return list(filter(lambda x:x,map(lambda x: globalLink(respURL,x),m)))
