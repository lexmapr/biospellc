import csv
import re
import os
import string
from collections import Counter
from difflib import get_close_matches
import biospellc.biospellc_helpers as h
from definitions import ROOT


class SpellChecker(object):

  def __init__(self, corpus_file_path):
    with open(corpus_file_path, "r", encoding="utf8") as file:
      lines = file.readlines()
      words = []
      for line in lines:
        words += re.findall(r"[\w+-]*\S", line.lower())

    self.vocabs = set(words)
    self.word_counts = Counter(words)
    total_words = float(sum(self.word_counts.values()))
    self.word_probas = {word: self.word_counts[word] / total_words for word in self.vocabs}

  def read_file(filename):
    with open(filename, "r", encoding="utf8") as file:
      lines = file.readlines()
      words = []
      for line in lines:
        words += re.findall(r'\w+', line.lower())

    return words

  def _level_one_edits(self, word):
    letters = string.ascii_lowercase
    splits = [(word[:i], word[i:]) for i in range(len(word) + 1)]
    deletes = [l + r[1:] for l,r in splits if r]
    swaps = [l + r[1] + r[0] + r[2:] for l, r in splits if len(r)>1]
    replaces = [l + c + r[1:] for l, r in splits if r for c in letters]
    inserts = [l + c + r for l, r in splits for c in letters]

    return set(deletes + swaps + replaces + inserts)

  def _level_two_edits(self, word):
    return set(e2 for e1 in self._level_one_edits(word)
               for e2 in self._level_one_edits(e1) if e2 in self.vocabs)

  def _level_three_edits(self, word):
    return set(e2 for e1 in self._level_two_edits(word)
               for e2 in self._level_two_edits(e1))

  def check(self, word):
    candidates = self._level_one_edits(word) or self._level_two_edits(word)  \
                 or self._level_three_edits(word) or [word]
    valid_candidates = [w for w in candidates if w in self.vocabs]
    return sorted([(c, self.word_probas[c]) for c in valid_candidates],
                  key=lambda tup: tup[1], reverse=True)

  def get_candidates (self, word):
      valid_candidates = []
      selected_matches = get_close_matches(word, self.vocabs, n=4, cutoff=0.65)
      for match in selected_matches:
          valid_candidate = (match,0.9)
          valid_candidates.append(valid_candidate)

      return valid_candidates



#Declaration of the output files
output_filePath = os.path.join(ROOT, "data",
                               './outputs/output_SpellCor_CinecaEvalSet2.tsv')
fw = open(output_filePath, 'w')

# Gets path of various dictionarries/corpus data
english_dictionary_filePath = os.path.join(ROOT, "data",'./inputs/generalized_english_dictionary.txt')
generalized_english_corpus_filePath = os.path.join(ROOT, "data",'./inputs/generalized_english_corpus.txt')
biomedical_corpus_terminological_filePath = os.path.join(ROOT, "data",'./inputs/biomedical_corpus.txt')
biomedical_corpus_full_filePath = os.path.join(ROOT, "data",'./inputs/biomedical_corpus_full.txt')

print("It is reading large corpora files. "
      "Please wait, it will take some time "
      "(currently appox. 5-7 minutes)......")

# Gets various dictionarries/corpus
sc_english_dictionary = SpellChecker(english_dictionary_filePath)
sc_english_corpus = SpellChecker(generalized_english_corpus_filePath)
sc_cineca_domain_term = SpellChecker(biomedical_corpus_terminological_filePath)
sc_cineca_domain = SpellChecker(biomedical_corpus_full_filePath)
#sc_cineca_domain_full = SpellChecker("./data/inputs/biomedical_corpus_full.txt")

# Gets all spelling mistake examples from resource in CSV file format
# and put in a dictionary to be used further
customized_spelling_correction_filePath = os.path.join(ROOT, "data",'./inputs/customized_spelling_correction.csv')

spelling_dict = {}
with open(customized_spelling_correction_filePath) as csvfile:
    ctr = 0
    read_csv_file = csv.reader(csvfile, delimiter=',')
    for row in read_csv_file:
        if ctr > 0:
            term = row[0]
            correction = row[1]
            spelling_dict[term.strip()] = (correction.strip()).lower()
        ctr += 1


def get_corpus_resource(corpus_filename):
    corpus_resource = SpellChecker(corpus_filename)

    return corpus_resource


#========== Main pipeline ==================

frequent_bigram_list = h.get_corpus_bigrams(biomedical_corpus_full_filePath)
input_text_filePath = os.path.join(ROOT, "data",'./inputs/'
                                                'input_spellingcorrection_cineca_evaluationdata.txt')
# Input  the word (could be compound word or phrase) to be spell-checked and corrected
words_tobe_checked = h.read_file_linewise(input_text_filePath)

missing_vocab = set()
for word in words_tobe_checked:
    #testing_word = testing_word.strip()
    print("tested word - "+word)
    fw.write("\n" + word + "\t")
    word = h.pre_processing(word)
    returned_prob = []

    final_correction_status = "Same"
    correction_status = set()
    # correction_status.add("Same")
    processed_status = "Same"
    corrected_word = ""
    valid_candidates_for_refinement = []
    token_list = h.split_phrase(word)

    for token in token_list:
        token = token.lower()
        if token in sc_english_dictionary.vocabs or \
                   h. singularize_token(token) in \
                    sc_english_dictionary.vocabs or \
            h.is_number(token) or h.is_date(token):
            #corrected_word = token
            corrected_word += " " + (str(token))
            correction_status.add("Same- Token Exists in English "
                                  "Vocabulary [" + token + "]")
        elif token in sc_cineca_domain_term.vocabs \
                or h.singularize_token(token) in sc_cineca_domain_term.vocabs:
            # corrected_word = token
            corrected_word += " " + (str(token))
            correction_status.add("Same- Token Exists in Domain-Specific"
                                  " Terminological-Corpus [" + token + "]")

        elif (token.lower() in spelling_dict.keys()):
            corrected_token = h.customized_spell_correct(token, spelling_dict)
            corrected_word += " " + (str(corrected_token))
            correction_status.add("Corrected- Using Customized "
                                  "Lookup-Table [" + token + " ==> " + corrected_token + "]")

        else:
            valid_candidates_domain_term = sc_cineca_domain_term.check(token)
            valid_candidates_dictionary = sc_english_dictionary.check(token)
            valid_candidates_multiedit = sc_cineca_domain.get_candidates(token)

            if valid_candidates_domain_term:
                valid_candidates_for_refinement = valid_candidates_for_refinement + \
                                                  valid_candidates_domain_term
                corrected_token = str(valid_candidates_domain_term[0][0])
                corrected_word += " " + (str(corrected_token))
                #returned_prob.append(str(valid_candidates_domain_term))
                correction_status.add("Corrected - Using Single-Edit"
                                      " Probable Candidates in Domain "
                                      "Corpus [" + token + " ==> " + corrected_token + "]")
            elif valid_candidates_dictionary:
                valid_candidates_for_refinement = valid_candidates_for_refinement + valid_candidates_dictionary
                corrected_token = str(valid_candidates_dictionary[0][0])
                corrected_word += " " + (str(corrected_token))
                #returned_prob.append(str(valid_candidates_dictionary))
                correction_status.add("Corrected - Using Single-Edit"
                                      " Probable Candidates in English "
                                      "Corpus [" + token + " ==> " + corrected_token + "]")
            elif valid_candidates_multiedit:
                valid_candidates_for_refinement = valid_candidates_for_refinement + \
                                                  valid_candidates_multiedit
                #returned_prob.append(str(valid_candidates_multiedit))
                missing_vocab.add(str(token))
                corrected_token = str(valid_candidates_multiedit[0][0])
                corrected_word += " " + (str(corrected_token))
                correction_status.add("Corrected - Using "
                                      "MultiEdit Probable Candidates [" + token +
                                      " ==> " + corrected_token + "]")
            else:
                corrected_word += " " + (str(token))
                correction_status.add("Same - Word/Token Not found in "
                                      "English Dictionary or Domain "
                                      "Corpus Or Not Corrected [" + token + "]")

    corrected_word = corrected_word.strip()
    if "Corrected" in str(correction_status):
        processed_status = "Changed"
        returned_prob.append(str(valid_candidates_for_refinement))

    if processed_status == "Changed" and len(token_list) >= 2 \
            and len(valid_candidates_for_refinement) > 0:
        refined_word = h.get_refined_output(word, corrected_word,
                                            valid_candidates_for_refinement, frequent_bigram_list)
        if refined_word != corrected_word:
            correction_status.add("Refined/Validated Correction - "
                                  "Context-aware post-processing "
                                  " [" + corrected_word + " ==> " + refined_word + "]")
            corrected_word = refined_word#refined_word

                # here will come the module for optimal refinement

    print("corrected_word - " + corrected_word)
    fw.write(str(returned_prob))
    fw.write("\t" + corrected_word + "\t" + str(processed_status)+ "\t" + str(correction_status))

fw.write("\n\n" + str(missing_vocab))
fw.close()