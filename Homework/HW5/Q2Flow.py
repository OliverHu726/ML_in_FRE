from metaflow import FlowSpec, step, Parameter, IncludeFile, current
from datetime import datetime
import pandas as pd
import numpy as np
import string
from collections import Counter
from collections import defaultdict
import os

assert os.environ.get('METAFLOW_DEFAULT_DATASTORE', 'local') == 'local'
assert os.environ.get('METAFLOW_DEFAULT_ENVIRONMENT', 'local') == 'local'

class dh3517_Q2_flow(FlowSpec):

    TRAIN_FILE = IncludeFile(
        'dataset',
        help='Train data file',
        is_text=False,
        default='/Users/oliverhu/Desktop/Q2/src/graham.txt')

    @step
    def start(self):
        """
        Start up and print out some info to make sure everything is ok metaflow-side
        """
        print("Starting up at {}".format(datetime.utcnow()))
        # debug printing - this is from https://docs.metaflow.org/metaflow/tagging
        print("flow name: %s" % current.flow_name)
        print("run id: %s" % current.run_id)
        print("username: %s" % current.username)
        self.next(self.load_data)

    @step
    def load_data(self):
        """
        Loading the training data
        """
        # some utils function
        def prepare_sentence(sentence: str):
            processed_sentence = pre_process_sentence(sentence)
            return tokenize_sentence(processed_sentence)
        def pre_process_sentence(sentence: str):
            lower_sentence = sentence.lower()
            exclude = set(string.punctuation)
            return ''.join(ch for ch in lower_sentence if ch not in exclude)
        def tokenize_sentence(sentence: str):
            return sentence.split()
        def get_corpus_from_text_file(text_file: str):
            with open(text_file, 'r') as file:
                sentences = [_ for _ in [s.strip() for s in file.read().replace(';', '.').split('.')] if _]
                # debug 
                print("{} raw sentences found in {}".format(len(sentences), text_file))
            # clean the sentences and remove empty ones
            cleaned_sentences = [_ for _ in [prepare_sentence(s) for s in sentences] if _]
            # debug 
            print("{} cleaned sentences from {}\n".format(len(cleaned_sentences), text_file))       
            return cleaned_sentences
        def get_all_words_from_dataset(dataset: list):
            # we first FLAT a list of list 
            # [['i', 'am', 'jacopo'], ['you', 'are', 'funny']] ->
            # ['i', 'am', 'jacopo', 'you', 'are', 'funny']
            # and feed the list to a counter object
            return [word for sentence in dataset for word in sentence]
        DATASETS = {}
        # We decide to use shakespeare text here to train the models
        # some shakespeare stuff, https://www.gutenberg.org/ebooks/author/65
        DATASETS['paul'] = get_corpus_from_text_file('/Users/oliverhu/Desktop/Q2/src/graham.txt')    
        cnt_corpus = DATASETS['paul']
        self.WORDS = Counter(get_all_words_from_dataset(cnt_corpus))
        # get words frequency
        self.WORDS.most_common(10)
        self.next(self.original_model, self.improved_model)

    @step
    def original_model(self):
        """
        This is the process of original correction model from professor's netebook
        """
        def P(word, N=sum(self.WORDS.values())): 
            "Probability of `word`."
            return self.WORDS[word] / N
        def correction(word): 
            "Most probable spelling correction for word."
            return max(candidates(word), key=P)
        def candidates(word): 
            "Generate possible spelling corrections for word."
            return (known([word]) or known(edits1(word)) or known(edits2(word)) or [word])
        def known(words): 
            "The subset of `words` that appear in the dictionary of WORDS."
            return set(w for w in words if w in self.WORDS)
        def edits1(word):
            "All edits that are one edit away from `word`."
            letters    = 'abcdefghijklmnopqrstuvwxyz'
            splits     = [(word[:i], word[i:])    for i in range(len(word) + 1)]
            deletes    = [L + R[1:]               for L, R in splits if R]
            transposes = [L + R[1] + R[0] + R[2:] for L, R in splits if len(R)>1]
            replaces   = [L + c + R[1:]           for L, R in splits if R for c in letters]
            inserts    = [L + c + R               for L, R in splits for c in letters]
            return set(deletes + transposes + replaces + inserts)
        def edits2(word): 
            "All edits that are two edits away from `word`."
            return (e2 for e1 in edits1(word) for e2 in edits1(e1))
        def unit_tests():
            assert correction('beause') == 'because'                # insert
            assert correction('srartupz') == 'startup'              # replace 2
            assert correction('bycycle') == 'bicycle'               # replace
            assert correction('inconvient') == 'inconvenient'       # insert 2
            assert correction('arrainged') == 'arranged'            # delete
            assert correction('peotry') =='poetry'                  # transpose
            assert correction('peotryy') =='poetry'                 # transpose + delete
            assert correction('word') == 'word'                     # known
            assert correction('quintessential') == 'quintessential' # unknown
            assert self.WORDS.most_common(1)[0][0] == 'the'
            assert P('trafalgar') == 0
            return 'unit_tests pass'
        def spelltest(tests, verbose=False):
            "Run correction(wrong) on all (right, wrong) pairs; report results."
            import time
            start = time.time()
            good, unknown = 0, 0
            n = len(tests)
            for right, wrong in tests:
                w = correction(wrong)
                good += (w == right)
                if w != right:
                    unknown += (right not in self.WORDS)
                    if verbose:
                        print('correction({}) => {} ({}); expected {} ({})'.format(wrong, w, self.WORDS[w], right, self.WORDS[right]))
            dt = time.time() - start
            print("Original Model")
            print('{:.0%} of {} correct ({:.0%} unknown) at {:.0f} words per second '
                .format(good / n, n, unknown / n, n / dt))
            result = {'correct': good / n, 'unknown': unknown / n, 'process rate': n / dt}
            return result
        def Testset(lines):
            "Parse 'right: wrong1 wrong2' lines into [('right', 'wrong1'), ('right', 'wrong2')] pairs."
            return [(right, wrong)
                    for (right, wrongs) in (line.split(':') for line in lines)
                    for wrong in wrongs.split()]
        print("Results of Original Model")
        print(unit_tests())
        self.test1_result = spelltest(Testset(open('/Users/oliverhu/Desktop/Q2/src/spell-testset1.txt'))) # Development set
        self.test2_result = spelltest(Testset(open('/Users/oliverhu/Desktop/Q2/src/spell-testset2.txt'))) # Final test set
        self.next(self.join)

    @step
    def improved_model(self):
        """
        The imporved model is developed on 2 ideas:
        1. Remove the corrections with different first letter with the input word, if there are correction with the same first letter.
        Example: Input 'lable' with corrections {'label', 'table'}, 'table' will be removed
        2. If all corrections have the same length with input word, we will calculate the euclidean distance on keyboard between changed letters,
        and pick the correction with the smallest distance
        3. Else, we will use the same algorithm with original model
        """
        def P(word, N=sum(self.WORDS.values())): 
            "Probability of `word`."
            return self.WORDS[word] / N
        def correction(word): 
            "Most probable spelling correction for word."
            if len(candidates(word)) > 1 and all(len(corr) == len(word) for corr in candidates(word)):
                # Calculate keyboard distance for each correction
                keyboard_distances = [(w, keyboard_distance(word, w)) for w in candidates(word)]
                # Select the correction with the smallest keyboard distance
                if keyboard_distances:
                    best_correction = min(keyboard_distances, key=lambda x: x[1])[0]
                    return best_correction
            elif len([w for w in candidates(word) if w[0] == word[0]]) >= 1 :
                possible_can = candidates(word)
                # Filltered candidates that has different first letter with input word
                # Example: 'table' will be removed if input is 'lable'
                filltered_can = [w for w in possible_can if w[0] == word[0]]
                return max(filltered_can, key=P)
            else:
                return max(candidates(word), key=P)
        def candidates(word): 
            "Generate possible spelling corrections for word."
            return known([word]) or known(edits1(word)) or known(edits2(word)) or [word]
        def keyboard_distance(word1, word2):
            "Calculate keyboard distance between two words."
            keyboard_layout = {
                'q': (0, 0), 'w': (0, 1), 'e': (0, 2), 'r': (0, 3), 't': (0, 4),
                'a': (1, 0), 's': (1, 1), 'd': (1, 2), 'f': (1, 3), 'g': (1, 4),
                'z': (2, 0), 'x': (2, 1), 'c': (2, 2), 'v': (2, 3), 'b': (2, 4),
                'y': (0, 5), 'u': (0, 6), 'i': (0, 7), 'o': (0, 8), 'p': (0, 9),
                'h': (1, 5), 'j': (1, 6), 'k': (1, 7), 'l': (1, 8),
                'n': (2, 5), 'm': (2, 6),}
            distance = 0
            for char1, char2 in zip(word1, word2):
                pos1, pos2 = keyboard_layout.get(char1, (0, 0)), keyboard_layout.get(char2, (0, 0))
                distance += ((pos1[0] - pos2[0]) ** 2 + (pos1[1] - pos2[1]) ** 2) ** 0.5
            return distance
        def known(words): 
            "The subset of `words` that appear in the dictionary of WORDS."
            return set(w for w in words if w in self.WORDS)
        def edits1(word):
            "All edits that are one edit away from `word`."
            letters    = 'abcdefghijklmnopqrstuvwxyz'
            splits     = [(word[:i], word[i:])    for i in range(len(word) + 1)]
            deletes    = [L + R[1:]               for L, R in splits if R]
            transposes = [L + R[1] + R[0] + R[2:] for L, R in splits if len(R)>1]
            replaces   = [L + c + R[1:]           for L, R in splits if R for c in letters]
            inserts    = [L + c + R               for L, R in splits for c in letters]
            return set(deletes + transposes + replaces + inserts)
        def edits2(word): 
            "All edits that are two edits away from `word`."
            return (e2 for e1 in edits1(word) for e2 in edits1(e1))
        def unit_tests():
            assert correction('beause') == 'because'                # insert
            assert correction('srartupz') == 'startup'              # replace 2
            assert correction('bycycle') == 'bicycle'               # replace
            assert correction('inconvient') == 'inconvenient'       # insert 2
            assert correction('arrainged') == 'arranged'            # delete
            assert correction('peotry') =='poetry'                  # transpose
            assert correction('peotryy') =='poetry'                 # transpose + delete
            assert correction('word') == 'word'                     # known
            assert correction('quintessential') == 'quintessential' # unknown
            assert self.WORDS.most_common(1)[0][0] == 'the'
            assert P('trafalgar') == 0
            return 'unit_tests pass'
        def spelltest(tests, verbose=False):
            "Run correction(wrong) on all (right, wrong) pairs; report results."
            import time
            start = time.time()
            good, unknown = 0, 0
            n = len(tests)
            for right, wrong in tests:
                w = correction(wrong)
                good += (w == right)
                if w != right:
                    unknown += (right not in self.WORDS)
                    if verbose:
                        print('correction({}) => {} ({}); expected {} ({})'.format(wrong, w, self.WORDS[w], right, self.WORDS[right]))
            dt = time.time() - start
            print("Improved Model")
            print('{:.0%} of {} correct ({:.0%} unknown) at {:.0f} words per second '
                .format(good / n, n, unknown / n, n / dt))
            result = {'correct': good / n, 'unknown': unknown / n, 'process rate': n / dt}
            return result
        def Testset(lines):
            "Parse 'right: wrong1 wrong2' lines into [('right', 'wrong1'), ('right', 'wrong2')] pairs."
            return [(right, wrong)
                    for (right, wrongs) in (line.split(':') for line in lines)
                    for wrong in wrongs.split()]
        print("Results of Improved Model")
        print(unit_tests())
        self.test1_result = spelltest(Testset(open('/Users/oliverhu/Desktop/Q2/src/spell-testset1.txt'))) # Development set
        self.test2_result = spelltest(Testset(open('/Users/oliverhu/Desktop/Q2/src/spell-testset2.txt'))) # Final test set
        self.next(self.join)

    @step
    def join(self, inputs):
        print('Comparison of Original and Improved model')
        print('Result of Original Model:')
        print(inputs.original_model.test1_result)
        print(inputs.original_model.test2_result)
        print(inputs.improved_model.test1_result)
        print(inputs.improved_model.test2_result)
        self.next(self.end)

    @step
    def end(self):
        print("All done at {}!\n See you, dragon boys!".format(datetime.utcnow()))
    
if __name__ == '__main__': 
    dh3517_Q2_flow()



