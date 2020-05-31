# imports
import spacy
import en_core_web_sm
nlp = en_core_web_sm.load()


# load wikipedia corpus and keep only the page texts (remove titles and blank lines)
text = open("wikipedia-first.txt").read().splitlines()[1::3]


# abridged version of the corpus for tests
short_text = text[:30]

# explore part of speech (pos) characteristics of sentences
for sentence in short_text:
    doc = nlp(sentence)
    for token in doc:
        print(token.text, token.pos_, token.tag_)
    print()
    print()
  
    
# sample result:
# April PROPN nsubj AUX ROOT -> entity
# is AUX ROOT AUX ROOT -> verb
# the DET det NOUN attr
# fourth ADJ amod NOUN attr
# month NOUN attr AUX ROOT -> class
# of ADP prep NOUN attr
# the DET det NOUN pobj
# year NOUN pobj ADP prep
# with ADP prep NOUN attr
# 30 NUM nummod NOUN pobj
# days NOUN pobj ADP prep
# . PUNCT punct AUX ROOT
    
  
# conclusion: some useful traits and also some issues
# the entity from which we want to identify the class is always placed before a verb AUX or VBZ
# as for the class, it is always found after the verb, most of the time identified as NOUN
# however, it cn be a composite noun (earth atmosphere), accompanied by adjectives (American)
# it can also be parasitised with abstract expressions (as indicated in the lab instructions) like "sort of)
# such abstract expressions are problematic because they can contain nouns which might be interpreted as the class


# explore now dep(dependencies)
for sentence in short_text:
    doc = nlp(sentence)
    for token in doc:
        print(token.text, token.pos_, token.dep_, token.head.pos_, token.head.dep_)
    print()
    print()
    
# sample result
# Andouille PROPN nsubj AUX ROOT -> entity
# is AUX ROOT AUX ROOT -> verb
# a DET det NOUN attr
# sort NOUN attr AUX ROOT
# of ADP prep NOUN attr
# pork NOUN compound NOUN pobj
# sausage NOUN pobj ADP prep
# . PUNCT punct AUX ROOT    
    
    
# the following holds
# the sequence "entity, verb, class" is verified
# the class always comes after a verb, so it can appear only after some token.pos_ = AUX or VERB
# the class is always a noun (as specified in the lab instructions); so token.pos_ = NOUN
# its dependency type is most of the time attr (attribute), but other types like pobj or dobj
# on peut alors proposer l'algo simple suivant

def extractor(content):
    doc = nlp(content)
    after_verb = False
    for token in doc:
        # if a verb is reached, switch after_verb to True
        if (token.pos_ == 'AUX') or (token.pos_ == 'VERB'):
            after_verb = True
            # print(token)
            # print('yes')
        # if a verb has been reached, and the token is a noun, and if dependency is NN, NNS or ATTR, return token
        if (after_verb == True) and (token.pos_ == 'NOUN') and (token.dep_ == 'attr' or token.dep_ == 'pobj' or token.dep_ == 'dobj'):
            return token.text
        # if no correspondance is found, do not return anything
        pass
    
# explore now dep(dependencies)
for sentence in short_text:
    print(sentence)
    print(extractor(sentence))
print()
print()        
    
# sample result
# April is the fourth month of the year with 30 days.
# month
# August is the eighth month of the year.
# month
# The word art is used to describe some activities or creations of human beings that have importance to the human mind, regarding an attraction to the human senses.
# activities
# A is 1st letter of the alphabet.
# letter
# Air means Earth's atmosphere.
# atmosphere
# Spain is divided in 17 parts called autonomous communities.
# parts
# Alan Mathison Turing  was an English mathematician and computer scientist.
# scientist
# Alanis Nadine Morissette  is a Grammy Award-winning Canadian-American singer and songwriter.
# singer
# "Adobe Illustrator" is a computer program for making graphic design and illustrations.
# program
# Andouille is a sort of pork sausage.
# sort

# not bad but some misidentification results from abstract terms that are nouns (e.g. "sort of sausage")
# as already discussed, it is necessary to ignore these nouns corresponding to no real designation
# to do this, the simplest way is to create a list of terms that mst be ignored
# one can then rewrite the function as


def is_abstract_noun(noun):
    abstract_noun_list = ['sort', 'sorts', 'type', 'types', 'kind', 'kinds', 'member', 'members', 'way', 'ways',
                          'form', 'forms', 'group', 'groups']
    if noun in abstract_noun_list:
        return True
    else:
        return False


def extractor(content):
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


# explore now dep(dependencies)
for sentence in short_text:
    print(sentence)
    print(extractor(sentence))



# sample result: 
# April is the fourth month of the year with 30 days.
# month
# August is the eighth month of the year.
# month
# The word art is used to describe some activities or creations of human beings that have importance to the human mind, regarding an attraction to the human senses.
# activities
# A is 1st letter of the alphabet.
# letter
# Air means Earth's atmosphere.
# atmosphere
# Spain is divided in 17 parts called autonomous communities.
# parts
# Alan Mathison Turing  was an English mathematician and computer scientist.
# scientist
# Alanis Nadine Morissette  is a Grammy Award-winning Canadian-American singer and songwriter.
# singer
# "Adobe Illustrator" is a computer program for making graphic design and illustrations.
# program
# Andouille is a sort of pork sausage.
# sausage
# Farming is the growing of crops or keeping of animals by people for food and raw materials.
# growing
# Arithmetic is what we call working with numbers.
# numbers
# Addition is the mathematical way of putting things together.
# things
# Australia is a continent in the Southern Hemisphere between the Pacific Ocean and the Indian Ocean.
# continent
# American English or U.S.
# None
# Aquaculture is the farming of fish, shrimp, and algae.
# farming
# An abbreviation is a shorter way to write a word or phrase.
# word
# In many religions, an angel is a good spirit.
# spirit
# Ad hominem is a Latin term.
# term
# Native Americans  are those people who were in North America, Central America, South America, and the Caribbean Islands when the Europeans came there.
# people
# An apple is a kind of fruit.
# fruit
# People use the term Abrahamic Religion for a number of religions that recognise Abraham as an important person.
# term
# Algebra is a part of mathematics  that helps show the general links between numbers and math operations  used on the numbers.
# part
# In other words, you can say more generally what an idiom or metaphor says: for example, "battle of the sexes" is both a metaphor and an idiom that suggests "love as war".
# example
# An atom is the most simple type of particle that makes up matter.
# particle
# Astronomy is the study of planets, stars, galaxies, and other objects found in outer space.
# study
# Architecture is the design of structures; how they are built.
# design
# Anatomy is the study of the bodies of living beings .
# study
# An asteroid is like a planet, but smaller.
# planet
# Afghanistan  is a country located in South Asia.
# country

# that sounds good enough for me, I keep it







