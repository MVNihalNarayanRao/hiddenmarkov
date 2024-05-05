from collections import Counter
import re
import string
import sys
from hmmlearn import hmm

# Preprocessing corpus for unigrams and getting the words and their frequencies using Counter
def get_unigram_counts(filepath):
    filepath = filepath
    with open(filepath) as fp:
        sentences = []
        for line in (fp):
            # Adding <s> start and stop </s> in every sentence.
            line = '<s> ' + line + ' </s>'

            # Lower casing the sentences and splitting words over spaces
            words = line.lower().split()

            # Removing Punctuation
            sentence = [x for x in words if x not in (string.punctuation + '--' + '...' + '-' + '?')]
            sentences.extend(sentence)
    # Getting frequencies of words in Dictionary
    unigram_counts = Counter(sentences)
    return unigram_counts, len(sentences)

# Preprocessing the questions .txt file
def get_questions(Q_filepath):
    Q_filepath = Q_filepath
    with open(Q_filepath) as fp:
        questions = []
        for line in (fp):
            # Separating the candidate words at the end of each question
            words = line.replace('/', " ").lower().split()
            sentence = [x for x in words if x not in (string.punctuation + '--' + '...' + '-' + '?')]
            questions.append(sentence)
    return questions

def train_hmm(corpus):
    states = list(set(corpus))
    hmm_model = hmm.MultinomialHMM(n_components=len(states))
    hmm_model.startprob_ = [1 / len(states)] * len(states)
    
    # Transform unigram counts to a list of observations
    observation_seq = [[word] * count for word, count in corpus.items()]
    observations = [item for sublist in observation_seq for item in sublist]
    X = [[states.index(obs)] for obs in observations]
    
    hmm_model.fit(X)
    return hmm_model, states

def predict_hmm(hmm_model, states, questions):
    predictions = []
    for question in questions:
        obs_idx = [states.index(word) for word in question[:-2]]
        # Predicting the next word based on the observed sequence
        next_word_prob = hmm_model.predict_proba([obs_idx])
        predicted_word_idx = next_word_prob.argmax(axis=1)[0]
        predicted_word = states[predicted_word_idx]
        predictions.append(predicted_word)
    return predictions

def main():
    # Getting file paths from the command line arguments
    filepath = 'corpus.txt'  # sys.argv[1]
    Q_filepath = 'questions.txt'  # sys.argv[2]

    print("\n----------------- Working..." + "\t Please Wait... -----------------")

    # Getting Unigram Counts
    unigram_counts = []
    unigram_counts, total_word_count_in_corpus = get_unigram_counts(filepath)

    # Getting preprocessed Questions
    questions = get_questions(Q_filepath)

    # Training HMM on the corpus
    hmm_model, states = train_hmm(unigram_counts)

    # Predicting using HMM
    predictions = predict_hmm(hmm_model, states, questions)

    # Printing results
    print("\n Results for HMM Model\n")
    for i, prediction in enumerate(predictions):
        strn = ' '.join(questions[i][:-2])
        print(f"[{i + 1}] {strn.replace('____', prediction)} --- Chosen word: {prediction}")
    print("\n --------------------------------------------------------------------------- \n")


if __name__ == '__main__':
    main()
