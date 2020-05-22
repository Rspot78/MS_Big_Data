# Written with <3 by Julien Romero

import itertools
import hashlib
from sys import argv
import sys
if (sys.version_info > (3, 0)):
    from urllib.request import urlopen
    from urllib.parse import urlencode
else:
    from urllib2 import urlopen
    from urllib import urlencode


NAME = "dupond".lower()
# This is the correct location on the moodle
ENCFILE = "/home/dupond/inf344/" + NAME + ".enc"
# If you run the script on your computer: uncomment and fill the following
# line. Do not forget to comment this line again when you submit your code
# on the moodle.
# ENCFILE = "PATH TO YOUR ENC FILE"



class Crack:
    """Crack The general method used to crack the passwords"""

    def __init__(self, filename, name):
        """__init__
        Initialize the cracking session
        :param filename: The file with the encrypted passwords
        :param name: Your name
        :return: Nothing
        """
        self.name = name.lower()
        self.passwords = get_passwords(filename)

    def check_password(self, password):
        """check_password
        Checks if the password is correct
        !! This method should not be modified !!
        :param password: A string representing the password
        :return: Whether the password is correct or not
        """
        password = str(password)
        cond = False
        if (sys.version_info > (3, 0)):
            cond = hashlib.md5(bytes(password, "utf-8")).hexdigest() in \
                self.passwords
        else:
            cond = hashlib.md5(bytearray(password)).hexdigest() in \
                self.passwords
        if cond:
            print("You found the password: " + password)
            return True
        return False
    
    
    # function to convert tuples into strings
    # inspired from https://www.geeksforgeeks.org/python-program-to-convert-a-tuple-to-a-string
    def tuple_to_string(self, tupl): 
        str =  ''.join(tupl) 
        return str
    
    
    # function to create list of words from txt file
    # inspired from https://www.w3schools.com/python/python_file_open.asp
    def word_list(self, file_path):
        word_list = []
        file = open(file_path, "r")
        for word in file:
            # suppress last character as it is a line break character
            word_list.append(word[:-1])
        file.close()
        return word_list
    
    
    # function to convert word to leet form
    def word_to_leet(self, word):
        leet = ""
        for letter in word:
            if letter == 'e':
                leet += '3'
            elif letter == 'l':
                leet += '1'
            elif letter == 'a':
                leet += '@'
            elif letter == 'o':
                leet += '0'
            else:
                leet += letter
        return leet 
    
    
    # function to add up to three hyphens at random positions
    def add_hyphen(self, word):
        """
        generates all possible hyphens combinations
        smartly inspired from https://stackoverflow.com/questions/5254445/add-string-in-a-certain-position-in-python
        """
        # get word length
        word_length = len(word)
        # if single letter, terminate
        if word_length == 1:
            return 'a'
        # otherwise, create empty list to be filled    
        word_list = []
        # first check for one hyphen
        for i in range(1, word_length):
            word_list.append(word[:i] + '-' + word[i:])
        # if size is at least 3, test for two hyphens
        if word_length>2:
            for i in range(1, word_length):
                for j in range(i,word_length):
                    word_list.append(word[:i] + '-' + word[i:j] + '-' + word[j:])
        # if size is at least 4, test for three hyphens
        if word_length>3:
            for i in range(1,word_length):
                for j in range(i,word_length):
                    for k in range(j,word_length):
                        word_list.append(word[:i] + '-' + word[i:j] + '-' + word[j:k] + '-' + word[k:])    
        return word_list

    
    def crack(self):
        """crack
        Cracks the passwords. YOUR CODE GOES BELOW.

        We suggest you use one function per question. Once a password is found,
        it is memorized by the server, thus you can comment the call to the
        corresponding function once you find all the corresponding passwords.
        """
        # self.bruteforce_digits()
        # self.bruteforce_letters()

        # self.dictionary_passwords()
        # self.dictionary_passwords_leet()
        # self.dictionary_words_hyphen()
        # self.dictionary_words_digits()
        # self.dictionary_words_diacritics()
        # self.dictionary_city_diceware()

        # self.social_google()
        # self.social_jdoe()
        # self.paste()

        
    def bruteforce_digits(self):
        """
        Longer passwords are harder to guess because they require a significantly longer running time. 
        Indeed, a password of length m and n possible elements for each character requires testing m**n combinations.
        As m gest large, m**n, gets large quickly, rendering the computational cost prohibitive.
        Here with m=9 and n=10 (10 possible digits), there is a maximum of 3 486 784 401 combinations to test.
        """
        # create a list of digits on which to test combinations
        digits = '0123456789'
        # we want to test any length between 1 and 9 digits
        for length in range(1,10):
            # use itertool to find all possible combinations
            for tuple in itertools.product(digits, repeat=length):
                password = self.tuple_to_string(tuple)
                self.check_password(password)

                
    def bruteforce_letters(self):
        """
        Passwords with more different characters are harder to guess using brute-force method because the
        number of different characters n is the power of the number of combinations m**n
        As n gest large, m**n gets large extremely fast, rendering the computational cost prohibitive.
        Here with m=5 and n=52 (52 possible letters, mixed case), there is a maximum 
        of 2 220 446 049 250 313 080 847 263 336 181 640 625 combinations, much more than with the digits.
        This is excessively long and slow and hence does not constitute a good strategy!!
        """
        # create a list of letters (mixed case) on which to test combinations
        letters = 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ'
        # we want to test any length between 1 and 5 letters
        for length in range(1,6):
            # use itertool to find all possible combinations
            for tuple in itertools.product(letters, repeat=length):
                password = self.tuple_to_string(tuple)
                self.check_password(password)

    def dictionary_passwords(self):
        """
        check all passwords from a list of the 10k most common passwords. List is taken from:
        https://github.com/danielmiessler/SecLists/blob/master/Passwords/Common-Credentials/10k-most-common.txt
        I rename the file as 10k.txt
        """
        passwords = self.word_list("/home/dupond/inf344/10k.txt")
        for password in passwords:
            self.check_password(password)
        

    def dictionary_passwords_leet(self):
        """
        check all passwords from a list of the 10k most common passwords. List is taken from:
        https://github.com/danielmiessler/SecLists/blob/master/Passwords/Common-Credentials/10k-most-common.txt
        I rename the file as 10k.txt
        All words are then converted to leet form
        """
        passwords = self.word_list("/home/dupond/inf344/10k.txt")
        for password in passwords:
            leet = self.word_to_leet(password)
            self.check_password(leet)
        
        
    def dictionary_words_hyphen(self):
        """
        check random addition of hyphens in passwords
        uses the 20k most common words of English language List is from:
        https://github.com/first20hours/google-10000-english/blob/master/20k.txt
        """
        passwords = self.word_list("/home/dupond/inf344/20kenglish.txt")
        for password in passwords:
            # produce the list of added hyphens
            hyphen_words = self.add_hyphen(password)
            # then check words, one by one
            for word in hyphen_words:
                self.check_password(word)
        

    def dictionary_words_digits(self):
        pass
        
        

    def dictionary_words_diacritics(self):
        """
        uses the 20k most frequent French words and computes all possible replacements of diacritics characters
        """
        words = self.word_list("/home/dupond/inf344/10kfrench.txt")
        for word in words:
            letter_list = []
            # check each letter, get regular equivalent
            for letter in word:
                if letter == 'é':
                    letter_list.append('ée')
                elif letter == 'è':
                    letter_list.append('èe')
                elif letter == 'ù':
                    letter_list.append('ùu')
                elif letter == 'à':
                    letter_list.append('àa')
                elif letter == 'ç':
                    letter_list.append('çc')
                else:
                    letter_list.append(letter)
            string_letter_list = str(letter_list)[1:-1]
            string_to_eval = "[s for s in itertools.product(" + string_letter_list + ")]"
            tuple_list = eval(string_to_eval)
            for element in tuple_list:
                password = tuple_to_string(element)
                self.check_password(password)
        

    def dictionary_city_diceware(self):
        """
        take the names of all African capitals and computes dicewares
        """
        words = self.word_list("/home/dupond/inf344/capitals.txt")
        for password in words:
            self.check_password(password)       
        # diceware of size 2
        for word1 in words:
            for word2 in words:
                password = word1 + '-' + word2
                self.check_password(password)
        # diceware of size 3
        for word1 in words:
            for word2 in words:
                for word3 in words:
                    password = word1 + '-' + word2 + '-' + word3
                    self.check_password(password)        
        # diceware of size 4
        for word1 in words:
            for word2 in words:
                for word3 in words:
                    for word4 in words:
                        password = word1 + '-' + word2 + '-' + word3 + '-' + word4
                        self.check_password(password)         


    def social_google(self):
        """
        starts from known password 'Prometheus' and tries some transformations to find Doe's password
        """
        # start from known password
        word1 = 'Prometheus'
        # add two digits at the end
        for tuple in itertools.product(digits, repeat=2):
            word2 = word1 + self.tuple_to_string(tuple)
            self.check_password(word2)
            # add leets
            letter_list = []
            for letter in word2:
                if letter == 'e':
                    letter_list.append('e3')
                elif letter == 'o':
                    letter_list.append('o0')                    
                else:
                    letter_list.append(letter)
            string_letter_list = str(letter_list)[1:-1]
            string_to_eval = "[s for s in itertools.product(" + string_letter_list + ")]"
            tuple_list = eval(string_to_eval)
            for element in tuple_list:
                word3 = tuple_to_string(element)
                self.check_password(word3)
                # add random hyphens
                hyphen_words = self.add_hyphen(word3)
                # then check words, one by one
                for word4 in hyphen_words:
                    self.check_password(word4)

                    
    def social_jdoe(self):
        """
        takes key words about John Doe's life and combines them to find passwords
        """
        # list of likely words, given information
        words = ['John', 'Doe', 'john', 'doe', 'April', '25', '1978', '04251978', 'Shaker', 'Heights',\
                 'shaker', 'heights', 'Ohio', 'ohio', 'New', 'York', 'new', 'york', 'Cleveland', 'Indians',\
                 'cleveland', 'indians', 'sunday', 'Shakespeare', 'shakespeare']
        # then iterate to test combinations up to 4 words combined
        for length in range(1,5):
            # use itertool to find all possible combinations
            for tuple in itertools.product(words, repeat=length):
                password = self.tuple_to_string(tuple)
                self.check_password(password)
        

    def paste(self):
        """
        uses password of bryant.rivera@hotmail.com found on the web
        password obtained from https://throwbin.io/RvokH9z
        """
        password_from_paste = "Riseagainst1"
        self.check_password(password_from_paste)



def get_passwords(filename):
    """get_passwords
    Get the passwords from a file
    :param filename: The name of the file which stores the passwords
    :return: The set of passwords
    """
    passwords = set()
    with open(filename, "r") as f:
        for line in f:
            passwords.add(line.strip())
    return passwords


# First argument is the password file, the second your name


if __name__ == "__main__":
    crack = Crack(ENCFILE, NAME)

    crack.crack()
