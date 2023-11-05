import re
import numpy as np
import random
import sys
import pandas as pd
import time
from nltk.tokenize import sent_tokenize
from nltk.tokenize import word_tokenize
from nltk.corpus import words

# define a function that add start/end characters accordingly based on sentences and cleanup
def word_from_sentence(sentence_i):
    """
    Input -- Takes in a sentence.

    Output -- Edit and append in start and stop characters accordingly.
    -->> Remove all the punctuations and unecessary stuff.
    """
    words = word_tokenize(sentence_i)
    words = [w for word in words for w in word.split('-')]
    words = [x for x in words if len(x)>0]
    new_words = [re.sub('[.,;:&`\s+\[\]\^\(\)\*\-\_\'\"\?\!]', '', word.lower()) for word in words if not re.fullmatch('^[.,;:&`\s+\[\]\^\(\)\*\-\_\'\"\?\!]$',word)]
    new_words = [x for x in new_words if len(x)>0]
    try:  ### Trying to insert the Stop Character.
        if words[-1] == '.':
            new_words.append('<\s>')
    except ValueError:
        ### No periods in the sentence, carry on.
        print('No periods in sentence.')
        pass
    try:
        if words[0][0].isupper():
            new_words.insert(0,'<s>')   ### Inserting Start Character.  ## This seems to be wrong.
    except IndexError:
        pass
    return new_words

# define function to do bigram calculation
def bigrams_calculation(corpus, training_set):

    # placeholder for bigram matrix
    ngram = np.zeros((len(corpus),len(corpus)))

    # loop through each line
    for l in training_set:
        # split sentences for each line
        sentences = sent_tokenize(l)
        # loop through each sentences
        for sent in sentences:
            # clean up sentences by tokenizing and adding start and end characters accordingly
            word_seq = word_from_sentence(sent)
            word_seq = [w if w in corpus else '<unk>' for w in word_seq]

            # calculate bigram for the author
            for w in range(len(word_seq)-1):
                # get indices for current and next word
                wn_1 = corpus.index(word_seq[w])
                wn = corpus.index(word_seq[w+1])
                # incrzse the counter for the sequence
                ngram[wn_1, wn] += 1

    return ngram, None

# define function to do trigram calculation
def trigrams_calculation(corpus, training_set):

    # start from a bigram
    bigram, _ = bigrams_calculation(corpus,training_set)
    # find all nonzero count bigram combinations
    nonzero_index = np.argwhere(np.nonzero(bigram))

    # placeholder for trigram matrix
    ngram = np.zeros((nonzero_index.shape[0],len(corpus)))  ## row-wise is bigram pair, column wise is the third word.

    # loop through each line
    for l in training_set:

        # split sentences for each line
        sentences = sent_tokenize(l)
        # loop through each sentences
        for sent in sentences:
            # clean up sentences by tokenizing and adding start and end characters accordingly
            word_seq = word_from_sentence(sent)
            word_seq = [w if w in corpus else '<unk>' for w in word_seq]

            # calculate trigram
            for w in range(len(word_seq)-2):
                # get the current word and next word indices
                wn_2 = corpus.index(word_seq[w])
                wn_1 = corpus.index(word_seq[w+1])

                # only keep counts for the bigrams that has a nonzero count
                if wn_2 in nonzero_index[:,0]:
                    if wn_1 in nonzero_index[:,1]:
                        # the third word sequence
                        wn = corpus.index(word_seq[w+2])
                        # check the row index of wn-2wn-1 word sequence
                        where = np.where((nonzero_index[:,0]==wn_2) & (nonzero_index[:,1]==wn_1))  ### Might have errors
                        # add counter
                        ngram[where[0],wn] += 1

    return ngram, nonzero_index


# define function for ziptian smoothing
def ziptian_smoothing(bigram_matrix):
    """
    Good-Turing Discounting  -- Returns discounted-bigram count matrix.
    """
    smoothed_bigram_matrix = bigram_matrix
    [unique_counts,unique_counts_count] = np.unique(smoothed_bigram_matrix,return_counts=True)
    scaled_counts = np.zeros((1,len(unique_counts)-1))
    k_limit = 6

    for i in range(len(unique_counts)-1):
        scaled_counts[0][i] = (unique_counts[i]+1)*(unique_counts_count[i+1])/unique_counts_count[i]
    for j in range(k_limit):
        smoothed_bigram_matrix[smoothed_bigram_matrix==unique_counts[j]] = scaled_counts[0][j]
    return smoothed_bigram_matrix

# define function for laplace smoothing that add one to all cells
def laplace_smoothing(ngram):
    # Adding one to all the values

    smoothed_laplace_n_gram = ngram
    V = smoothed_laplace_n_gram.shape[1]
    wn_1_counts = np.sum(smoothed_laplace_n_gram,axis=1)
    smoothed_laplace_n_gram += 1  ### Adding one to all the values.
    for i in range(V):
        smoothed_laplace_n_gram[i,:] = smoothed_laplace_n_gram[i,:]/(wn_1_counts[i]+V)

    return smoothed_laplace_n_gram

# define function that does not smooth bigram and return as is
def no_smoothing(ngram):
    return ngram

# define function alphabetically sort dictionary for quick search in keys
def alphabetize_dictionary(dictionary):
    # create a quick search for dictionary
    dictionary = [w.lower() for w in dictionary]
    dictionary_dict = {}
    dictionary_keys = set(w[0] for w in dictionary)

    for k in dictionary_keys:
        words = []
        for w in dictionary:
            if w[0] == k:
                words.append(w)
        dictionary_dict[k] = words

    return dictionary_dict

# define a class for n-gram language model classifier
class n_gram_classifier:
    # initialize variables needed
    def __init__(self):
        # build an alphabetized nltk dictionary for faster search
        self.g_dictionary = alphabetize_dictionary(words.words())
        # placeholder for n-grams model
        self.ngram = {}
        # placeholder for non-zero bigram indices for trigram model
        self.bigram_nzero_idx = {}

    # define training procedure
    def train(self, development, ngrams_, smoothing_):
        """
            development : if 1 then have to create development/test set else have to read the test file
            ngrams_     : function for ngram model, either bigrams_calculation or trigrams_calculation
            smoothing_  : smoothing function to be applied, either laplace_smoothing or ziptian_smoothing or no_smoothing
        """
        # placeholder for development set
        self.development_set = {}

        # repeat for each author
        for author in self.author_list:
            print('Training for author:', author, '...')

            # open author file
            f_ = open(author + '_utf8.txt', encoding='utf-8')
            # read author file
            f_lines = f_.readlines()
            f_.close()
            # remove empty lines
            f_lines = [x for x in f_lines if len(x)>2]

            # split training-test if development set is not provided
            if development:
                # shuffle before split
                random.shuffle(f_lines)
                # split by 80:20
                training_set  = f_lines[:round(len(f_lines)*0.8)]
                self.development_set[author] = f_lines[round(len(f_lines)*0.8):]
            # if testing set is provided, train on everything
            else:
                training_set = f_lines

            # Compute the author's ngram matrix
            ngram, bigram_nzero_idx = ngrams_(self.corpus, training_set)
            ndim = ngram.ndim

            # Smooth ngram based on smoothing function input
            ngram = smoothing_(ngram)

            # Normalize the ngram
            if smoothing_ is not laplace_smoothing:
                ngram = np.nan_to_num(ngram/ngram.sum(axis=1)[:,None])

            # adjust for zero values to prevent underflow
            ngram[ngram==0] = 1e-6

            # Assign ngram matrix and bigram nonzero indices to the author's ngram
            self.ngram[author] = ngram
            self.bigram_nzero_idx[author] = bigram_nzero_idx

        self.ndim = ndim


    # define function to do author classification
    def predict(self, test_set=None):

        print('Entering testing stage ...')
        # if testing dataset is not given, get development set that was set aside during training
        if test_set == None:
            test_set = self.development_set

        # placeholder for probabilities, author name, and ground truth
        line_count = [len(v) for v in test_set.values()]
        prob = np.zeros((sum(line_count), len(self.author_list)))
        ground_truth = np.zeros(sum(line_count))

        # record ground truth from the testing set
        j = 0
        for key, value in test_set.items():
            # for each line in the development set
            for l in value:
                # get ground truth with author's index
                ground_truth[j] = self.author_list.index(key)
                j += 1

        # keep a counter for row value
        j = 0
        # loop through all testing set
        for value in test_set.values():
            for line in value:
                # loop through to compute proabability for each author
                for a, author in enumerate(self.author_list):
                    # tokenize word
                    word_sequence = word_from_sentence(line)
                    word_sequence = [w if w in self.corpus else '<unk>' for w in word_sequence]

                    # loop through every word
                    for w in range(len(word_sequence)-1):
                        # get indices for current and next word
                        wn_2 = self.corpus.index(word_sequence[w])
                        wn_1 = self.corpus.index(word_sequence[w+1])
                        # prediction for bigram
                        if self.ndim == 2:
                            # add to ngram probability for each ngram
                            prob[j, a] += np.log(self.ngram[author][wn_2, wn_1])

                        # prediction for trigram
                        if self.ndim == 3:
                            # stop if not enough word following the current word
                            if w > len(word_sequence)-3:
                                pass
                            # else compute probability
                            else:
                                # check if current word and next word exist in bigram
                                if wn_2 in self.bigram_nzero_idx[:,0]:
                                    if wn_1 in self.bigram_nzero_idx[:,1]:
                                        # the third word sequence
                                        wn = self.corpus.index(word_sequence[w+2])
                                        # check the row index of wn-2wn-1 word sequence
                                        where = np.where((self.bigram_nzero_idx[:,0]==wn_2) & (self.bigram_nzero_idx[:,1]==wn_1))
                                        # add log probability of wn-2wn-1wn/wn-2wn-1
                                        prob[j, a] += np.log(self.ngram[author][where[0],wn])
                                    # if the first 2 words sequence did not exist, set probability to approximately zero
                                    else:
                                        prob[j, a] += np.log(1e-6)
                j += 1

        # classify excerpts to author with the highest probability
        prediction = np.argmax(prob, axis=1)

        # if provided with testing set, output predicted author for each line
        if test_set != self.development_set:
            for line, p in enumerate(prediction):
                print(f'Line No {line : < 10}: {self.author_list[p]: >20}')

        # for each author, compute accuracy and report
        print('Results on dev set:')
        for i, key in enumerate(self.author_list):
            total_count = np.count_nonzero(ground_truth==i)
            correct_count = np.count_nonzero((ground_truth==i)&(prediction==i))
            if total_count == 0:
                accuracy = 0
            else:
                accuracy = correct_count/total_count
            print(f"{key: <10}  :  {accuracy:>10,.2%} {correct_count}/{total_count} correct")

        print(f"{'Overall': <10}  :  {np.count_nonzero(ground_truth==prediction)/len(ground_truth):>10,.2%} {np.count_nonzero(ground_truth==prediction)}/{len(ground_truth)} correct")

    # create a function that categorize words as unknown words based on nltk library, else return the known word
    def word_identifier(self, word):
        # check if the word exist in dictionary
        if re.fullmatch('^[a-zA-Z].*', word):
            if word in self.g_dictionary[word[0]]:
                # keep word as it is
                word = word
        else:
            # categorize word as an unknown word
            word = '<unk>'
        return word

    # build a corpus based on input to build bigram
    def build_corpus(self, author_file, test_file = None):

        # read in the list of author files
        text_files = open(author_file)
        # getting the names of author excerpts
        name_text_files = text_files.readlines()
        text_files.close()

        # get number of authors
        n_authors = len(name_text_files)
        # extract author names
        self.author_list = [x.split('_utf8.txt')[0] for x in name_text_files]
        # extract author excerpts file
        self.author_file_name = [x.split('\n')[0] for x in name_text_files]
        # initialize a global dictionary dependent on training dataset to build bigram matrix, start with unknown words, start and end characters
        self.corpus = ['<unk>', '<s>', '<\s>']
        # placeholder for unknown words
        self.unknown_words = []
        files = self.author_file_name

        if test_file is not None:
            print(test_file)
            files.append(test_file)

        print('Building corpus ...')
        # loop through all author's excerpts
        for f in files:
            # read in author's excerpts
            f_ = open(f, encoding='utf-8')
            f_lines = f_.readlines()
            f_.close()

            # remove lines that do not contain any characters
            f_lines = [x for x in f_lines if len(x)>2]

            # loop through each line
            for sent in f_lines:
                # tokenize words
                words = word_tokenize(sent)  ### how to deal with apostrophy?
                # split again if word contains hyphens
                words = [w for word in words for w in word.split('—')]
                # clean up words that contains punctuations
                new_words = [re.sub('[.,;:&`\s+\[\]\^\(\)\*\-\_\'\"\?\!]', '', word.lower()) for word in words if not re.fullmatch('^[.,;:&`\s+\[\]\^\(\)\*\-\_\'\"\?\!]$',word)]

                # loop through each new word
                for w in new_words:
                    w = w.lower()
                    # append to global dictionary if not exist
                    if w not in self.corpus:
                        # identify if it is a known word in nltk dictionary
                        if self.word_identifier(w) == '<unk>':
                            self.unknown_words.append(w)
                        else:
                            self.corpus.append(w)

        # Sorting the corpus alphabetically
        self.corpus.sort()
        self.unknown_words = list(set(self.unknown_words))

def main():
    # store filename from the user input
    filename = sys.argv[1]
    test_file_name = None

    # if not testing stage
    if len(sys.argv) == 2:
        # require development set
        development = 1
    # if in testing stage
    else:
        # package testing data set
        development = 0
        test_file_name = sys.argv[3]
        f = open(test_file_name, encoding='utf-8')
        f_lines = f.readlines()
        f_lines = [x for x in f_lines if len(x)>2]
        f.close()
        test_set = {test_file_name.split('_')[0]: f_lines}

    # initialize n-gram model
    model = n_gram_classifier()
    # train the model based on input from author list and apply desired smoothing function/ngram model accordingly
    model.build_corpus(filename, test_file_name)
    # 2 things you will need to change for testing sake:
    # a. bigrams_calculation  //  trigrams_calculation     ## Select either one for the different mode of calculation.
    # b. laplace_smoothing   //  ziptian_smoothing  //  no_smoothing
    model.train(development, trigrams_calculation, ziptian_smoothing)

    # predict testing/development set
    if development:
        model.predict()
    else:
        model.predict(test_set = test_set)





if __name__ == '__main__':
    main()
