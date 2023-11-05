# n-grams
Use n-grams language model to classify authors

This Python program 
- uses utf-8 for text file encoding
- cleans up texts
- employs nltk's library for dictionary, sentence, and word tokenization
- allows for 2 different language model:
    1. Bigram model (bigrams_calculation as input to the model.train() function)
    2. Trigram model (trigrams_calculation as input to the model.train() function)
- allows for 2 types of smoothing methods for bigram model only:
    1. Laplace Smoothing (laplace_smoothing as input to the model.train() function)
    2. Good-Turing Discounting (ziptian_smoothing as input to the model.train() function)
