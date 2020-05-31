'''Extracts type facts from a wikipedia file
usage: extractor.py wikipedia.txt output.txt

Every line of output.txt contains a fact of the form
    <title> TAB <type>
where <title> is the title of the Wikipedia page, and
<type> is a simple noun (excluding abstract types like
sort, kind, part, form, type, number, ...).

Note: the formatting of the output is already taken care of
by our template, you just have to complete the function
extractType below.

If you do not know the type of an entity, skip the article.
(Public skeleton code)'''



from parser import Parser
import sys
import re
import spacy
nlp = spacy.load("en_core_web_sm")

if len(sys.argv) != 3:
    print(__doc__)
    sys.exit(-1)

def is_abstract_noun(noun):
    abstract_noun_list = ['sort', 'sorts', 'type', 'types', 'kind', 'kinds', 'member', 'members', 'way', 'ways',
                          'form', 'forms', 'group', 'groups']
    if noun in abstract_noun_list:
        return True
    else:
        return False

def extractType(content):
    doc = nlp(content)
    after_verb = False
    for token in doc:
        # if a verb is reached, switch after_verb to True
        if (token.pos_ == 'AUX') or (token.pos_ == 'VERB'):
            after_verb = True
        # if a verb has been reached, and the token is a noun, and if dependency is attr, pobj or dobj, return token
        if (after_verb == True) and (token.pos_ == 'NOUN') and (token.dep_ == 'attr' or token.dep_ == 'pobj' or token.dep_ == 'dobj') and not is_abstract_noun(token.text):
            return token.text
        # if no correspondance is found, do not return anything
        pass

with open(sys.argv[2], 'w', encoding="utf-8") as output:
    for page in Parser(sys.argv[1]):
        typ = extractType(page.content)
        if typ:
            output.write(page.title + "\t" + typ + "\n")
            
            
            
            
            
            
            
            
            
      
            
            
            
            
            
            
            
            
            
            

