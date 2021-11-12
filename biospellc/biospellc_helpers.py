import nltk.chunk
import re
from nltk.tokenize import sent_tokenize, word_tokenize
import nltk
from nltk.collocations import *
import unicodedata
import inflection
from dateutil.parser import parse
from nltk.tokenize.treebank import TreebankWordDetokenizer
import csv


# Removes special characters
def remove_special_characters(text):
    text = re.sub('[^a-zA-Z0-9\s]', '', text)

    return text


# Converts tuple to list of string
def convert_tuple_to_str(input_tuple_list):
    output_list = []
    for value in input_tuple_list:
        value_str = str(value)
        value_str = value_str.replace("(", "")
        value_str = value_str.replace(")", "")
        value_str = value_str.replace("'", "")
        value_str = value_str.replace(",", "")
        output_list.append(value_str)

    return output_list


# Gets frequent tokens in a corpus given a minimum frequency
def get_frequent_token(input_corpus, min_frequency):
    input_corpus = remove_special_characters(input_corpus)
    tokens = nltk.word_tokenize(str(input_corpus))
    from nltk.corpus import stopwords
    stopset = set(stopwords.words('english'))
    # frequency_dict = collections.OrderedDict()
    frequency_dict = {}
    frequency_dict_selected = {}
    frequent_token_list = []
    for tkn in tokens:
        if tkn.lower() in frequency_dict.keys():
            tkn_freq = int(frequency_dict[tkn.lower()])
            tkn_freq += 1
            frequency_dict[tkn.lower()] = tkn_freq
        else:
            frequency_dict[tkn.lower()] = 1
    for key, value in frequency_dict.items():
        if int(value) >= min_frequency and str(key) not in stopset:
            frequent_token_list.append(key)
            frequency_dict_selected[key] = value

    return frequency_dict_selected




# Gets frequent ngrams in a corpus given a minimum frequency
def get_frequent_ngrams(input_corpus, ngram_size, min_ngram_frequency):

    # Declares measures to be  used in get_frequent_ngrams function
    bigram_measures = nltk.collocations.BigramAssocMeasures()
    trigram_measures = nltk.collocations.TrigramAssocMeasures()

    nltk_tokens = ""
    nltk_tokens = (word_tokenize(input_corpus))
    words = [w.lower() for w in nltk_tokens]
    print("Number of Words (corpus) : ", len(nltk_tokens))

    from nltk.corpus import stopwords
    stopset = set(stopwords.words('english'))
    filter_stops = lambda w: len(w) < 3 or w in stopset
    if ngram_size == 2:
        finder = BigramCollocationFinder.from_words(words)
    if ngram_size == 3:
        finder = TrigramCollocationFinder.from_words(words)
    if ngram_size == 4:
        finder = QuadgramCollocationFinder.from_words(words)

    finder.apply_word_filter(filter_stops)
    finder.apply_freq_filter(min_ngram_frequency)
    frequent_ngram_list = finder.nbest(bigram_measures.raw_freq, 25000000)
    frequent_gram_list_str = convert_tuple_to_str(frequent_ngram_list)

    return frequent_gram_list_str


# To remove the accented characters, if any
def remove_accented_chars(text):
    text = unicodedata.normalize('NFKD', text).encode\
        ('ascii', 'ignore').decode('utf-8', 'ignore')

    return text


# Performs spell correction based on customized lexicon
def customized_spell_correct (word, spelling_dict):
    # Customized misspellings are dealt with  here
    corrected_word = word.lower()
    if (word in spelling_dict.keys()):  # spelling mistakes taken care of
        corrected_word = spelling_dict[word.lower()]

    return corrected_word


# Makes a list of tokens from a phrase
def split_phrase(phrase):
  new_phrase_list = phrase.split(' ')

  return new_phrase_list


# Returns a difference of elements between two lists
def list_diff(list1, list2):
    out = []
    for ele in list1:
        if not ele in list2:
            out.append(ele)
    return out


# Gets the dictionary of next tokens for words in a phrase
def get_next_token (phrase):
    next_token_dict = {}
    word_list = split_phrase(phrase)
    wordlist_iterator = iter(word_list)
    next(wordlist_iterator)
    for word in word_list[:-1]:
        key = word
        value = next(wordlist_iterator)
        next_token_dict[key] = value

    return next_token_dict

# Example run - commented ===

# phrase = "First Second Third Fourth Fifth Sixth"
# next_token_dict = get_next_token(phrase)
# print (str(next_token_dict))
# word_list = split_phrase(phrase)
# print (str(word_list))
# for word in word_list:
#     print(word)



# Another example run - commented ===

# original_phrase = "yeat infetion"
# corrected_phrase = "heat infection"
# token_list_original = split_phrase(original_phrase)
# token_list_corrected = split_phrase(corrected_phrase)
# changed_word = list_diff (token_list_original, token_list_corrected)
# print(str(changed_word))
# valid_candidates = [('heat', 3.1506829893050066e-06), ('yeast', 2.363012241978755e-06), ('meat', 1.5753414946525033e-06)]
#
# for tuple in valid_candidates:
#     element_one = tuple[0]
#     print("element_one " + element_one)



def read_file_linewise(filename):
  with open(filename, "r", encoding="utf8") as file:
    lines = file.readlines()
    words = []
    for line in lines:
      words.append(str(line).lower().replace("\n",""))

  return words

def split_phrase(phrase):
  new_phrase_list = phrase.split(' ')

  return new_phrase_list


def singularize_token(tkn):
    lemma = inflection.singularize(tkn)

    return lemma


# Determines  whether a string is a number (Used for Cardinal-Ordinal Tagging)
def is_number(inputstring):
    try:
        float(inputstring)
        return True
    except ValueError:
        pass
    try:
        import unicodedata
        unicodedata.numeric(inputstring)
        return True
    except (TypeError, ValueError):
        pass
    return False


# Determines  whether a string is a date or day (Used for DateOrDay Tagging)
def is_date(inputstring):
    try:
        parse(inputstring)
        return True
    except (ValueError, OverflowError):
        return False


# Gets refined output
def get_refined_output(original_phrase, corrected_phrase, valid_candidates,frequent_bigram_list):
    valid_candidate_words = []
    refined_word = ""
    flag = False
    token_list_original = split_phrase(original_phrase)
    token_list_corrected = split_phrase(corrected_phrase)
    changed_word_list = list_diff(token_list_original, token_list_corrected)
    if len(token_list_original) >= 2 and len(changed_word_list)>0 and len(valid_candidates)>1:
        changed_word = changed_word_list[0]
        #gets list of candidates by iterate of tuple
        changed_word_index = token_list_original.index(changed_word)
        list_end = len(token_list_original) - changed_word_index

        if changed_word_index != 0:
            prev_idx = changed_word_index - 1
            prev_word = token_list_original[prev_idx]
            for tuple in valid_candidates:
                element_one = tuple[0]
                candidate_bigram = prev_word + " " + str(element_one)
                if candidate_bigram in frequent_bigram_list:
                    refined_word = str(element_one)
                    token_list_original[changed_word_index] = refined_word
                    flag = True

        if list_end != 1:
            next_idx = changed_word_index + 1
            next_word = token_list_original[next_idx]
            for tuple in valid_candidates:
                element_one = tuple[0]
                #valid_candidate_words.append(element_one)
                candidate_bigram = str(element_one)+ " "+next_word
                if candidate_bigram in frequent_bigram_list:
                    refined_word = str(element_one)
                    token_list_original[changed_word_index] = refined_word
                    flag = True

    if flag == True:
        refined_output = TreebankWordDetokenizer().detokenize(token_list_original)
        return refined_output
    else:
       refined_output = corrected_phrase
       return refined_output



def customized_spell_correct (word, spelling_dict):
    # Customized misspellings are dealt with  here
    corrected_word = word.lower()
    if (word in spelling_dict.keys()):  # spelling mistakes taken care of
        corrected_word = spelling_dict[word.lower()]

    return corrected_word




def get_corpus_bigrams(corpus_filename):
    with open(corpus_filename, 'r', encoding="utf-8", errors='ignore') as f:
        input_data = f.read()
    frequent_bigram_list = get_frequent_ngrams(input_data, 2, 1)

    return frequent_bigram_list


def pre_processing(text):
    text = re.sub('[^a-zA-Z0-9-.%()\s]', ' ', text)
    text = text.strip()

    return text

