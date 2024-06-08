import collections
import re
import string
import sys


def preprocess_corpus_for_unigrams(file_path):
    """Process the corpus to generate unigrams and their frequencies."""
    with open(file_path, 'r') as file:
        tokens = []
        for line in file:
            line = '<s> ' + line.strip() + ' </s>'
            line = line.lower().split()
            line = [word for word in line if word not in string.punctuation]
            tokens.extend(line)
    unigram_freq = collections.Counter(tokens)
    return unigram_freq, len(tokens)


def preprocess_questions(question_file_path):
    """Process the questions to extract candidate words."""
    with open(question_file_path, 'r') as file:
        question_list = []
        for line in file:
            line = re.sub(r'/', " ", line.strip().lower())
            line = [word for word in line.split() if word not in string.punctuation]
            question_list.append(line)
    return question_list


def preprocess_corpus_for_bigrams(file_path):
    """Process the corpus to generate bigrams and their frequencies."""
    with open(file_path, 'r') as file:
        tokens = []
        for line in file:
            line = '<s> ' + line.strip() + ' </s>'
            line = line.lower().split()
            line = [word for word in line if word not in string.punctuation]
            tokens.extend(line)

        bigrams = [tokens[i] + " " + tokens[i + 1] for i in range(len(tokens) - 1)]
    bigram_freq = collections.Counter(bigrams)
    return bigram_freq, len(bigrams)


def calculate_bigram_probabilities(bigram_freq, unigram_freq):
    """Calculate bigram probabilities with and without smoothing."""
    bigram_prob = {bigram: freq / unigram_freq[bigram.split()[0]] for bigram, freq in bigram_freq.items()}
    bigram_prob_smoothed = {bigram: (freq + 1) / (unigram_freq[bigram.split()[0]] + len(unigram_freq))
                            for bigram, freq in bigram_freq.items()}
    return bigram_prob, bigram_prob_smoothed


def compute_sentence_probability(sentence, bigram_prob):
    """Compute the probability of a sentence given the bigram probabilities."""
    words = sentence.split()
    prob = 1.0
    for i in range(len(words) - 1):
        bigram = words[i] + " " + words[i + 1]
        prob *= bigram_prob.get(bigram, 0)
    return prob


def bigram_model(questions, bigram_prob, unigram_freq, correct_answers):
    """Predict missing words in questions using bigram model."""
    correct_count = 0
    results = []

    for q_index, question in enumerate(questions):
        sentence_prefix = ' '.join(question[:-2])
        first_option = '<s> ' + sentence_prefix.replace('____', question[-2]) + ' </s>'
        second_option = '<s> ' + sentence_prefix.replace('____', question[-1]) + ' </s>'

        first_prob = compute_sentence_probability(first_option, bigram_prob)
        second_prob = compute_sentence_probability(second_option, bigram_prob)

        if first_prob > second_prob:
            chosen_word = question[-2]
        else:
            chosen_word = question[-1]

        if chosen_word == correct_answers[q_index]:
            correct_count += 1

        results.append((sentence_prefix.replace('____', chosen_word), chosen_word, max(first_prob, second_prob)))

    return results, correct_count


def unigram_model(questions, unigram_freq, total_tokens, correct_answers):
    """Predict missing words in questions using unigram model."""
    correct_count = 0
    results = []

    for q_index, question in enumerate(questions):
        first_option_prob = unigram_freq[question[-2]] / total_tokens
        second_option_prob = unigram_freq[question[-1]] / total_tokens

        if first_option_prob > second_option_prob:
            chosen_word = question[-2]
        else:
            chosen_word = question[-1]

        if chosen_word == correct_answers[q_index]:
            correct_count += 1

        sentence_prefix = ' '.join(question[:-2])
        results.append((sentence_prefix.replace('____', chosen_word), chosen_word, max(first_option_prob, second_option_prob)))

    return results, correct_count


def main():
    corpus_path = 'corpus.txt'
    questions_path = 'questions.txt'
    correct_answers = ['whether', 'through', 'piece', 'court', 'allowed', 'check', 'hear', 'cereal', 'chews', 'sell']

    print("\nProcessing... Please Wait...\n")

    unigram_freq, total_tokens = preprocess_corpus_for_unigrams(corpus_path)
    bigram_freq, total_bigrams = preprocess_corpus_for_bigrams(corpus_path)
    questions = preprocess_questions(questions_path)

    print("\nResults for UNIGRAM Model\n")
    unigram_results, unigram_accuracy = unigram_model(questions, unigram_freq, total_tokens, correct_answers)
    for i, (sentence, word, prob) in enumerate(unigram_results):
        print(f"[{i + 1}] {sentence} --- Chosen word: {word} having probability {prob}")
    print(f"\nAccuracy of Unigram Model: {unigram_accuracy} out of {len(questions)}")

    print("\nResults for BIGRAM Model\n")
    bigram_prob, bigram_prob_smoothed = calculate_bigram_probabilities(bigram_freq, unigram_freq)
    bigram_results, bigram_accuracy = bigram_model(questions, bigram_prob, unigram_freq, correct_answers)
    for i, (sentence, word, prob) in enumerate(bigram_results):
        print(f"[{i + 1}] {sentence} --- Chosen word: {word} having probability {prob}")
    print(f"\nAccuracy of Bigram Model: {bigram_accuracy} out of {len(questions)}")

    print("\nResults for BIGRAM with Smoothing Model\n")
    bigram_smoothed_results, bigram_smoothed_accuracy = bigram_model(questions, bigram_prob_smoothed, unigram_freq, correct_answers)
    for i, (sentence, word, prob) in enumerate(bigram_smoothed_results):
        print(f"[{i + 1}] {sentence} --- Chosen word: {word} having probability {prob}")
    print(f"\nAccuracy of Bigram with Smoothing Model: {bigram_smoothed_accuracy} out of {len(questions)}")


if __name__ == '__main__':
    main()
