'''Extracts mails addresses
usage: extractor.py input.txt output.txt

Every line of output.txt contains a single mail address.
Note: the formatting of the output is already taken care of
by our template, you just have to complete the function
extractMail below.

(Public skeleton code)'''

import sys
import re

if len(sys.argv) != 3:
    print(__doc__)
    sys.exit(-1)

def extractMail(content):
    # define email adress in the dullest way: something between two whitespaces with a '@' somewhere in the string
    temp1 = re.findall('\S+@\S+', content)
    # do some cleaning: get rid of possible undesirable characters at beginning/end of string
    temp3 = []
    for element in temp1:
        temp2 = element.strip("<>(){}[]~#-|_=+")
        # check that email adress is not empty (nothing before or after '@')
        if (temp2[0] != '@') and (temp2[-1] != '@'):
            temp3.append(temp2)
    return temp3

inputContent = ""
with open(sys.argv[1], 'r', encoding="utf-8") as input:
    inputContent = input.read()
    
with open(sys.argv[2], 'w', encoding="utf-8") as output:
    for mail in extractMail(inputContent):
        output.write(mail + "\n")


    


























