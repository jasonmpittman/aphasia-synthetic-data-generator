__author__ = "Jason M. Pittman"
__copyright__ = "Copyright 2025"
__credits__ = ["Jason M. Pittman"]
__license__ = "Apache License 2.0"
__version__ = "0.1.0"
__maintainer__ = "Jason M. Pittman"
__status__ = "Research"

"""


Lexical richness measures derived from:
Yang, Y., & Zheng, Z. (2024). A Refined and Concise Model of Indices for Quantitatively Measuring Lexical Richness of Chinese University Students' EFL Writing. Contemporary Educational Technology, 16(3).
"""

import os
import re
import csv
import json
import random
import argparse
from collections import Counter
from dataclasses import dataclass

import nltk
from nltk.tokenize import word_tokenize
#nltk.download()

@dataclass
class PartsOfSpeech():
    #   content words
    nouns: list
    noun_frequency: int
    verbs: list
    verb_frequency: int
    adverbs: list
    adverb_frequency: int
    adjectives: list
    adjective_frequency: int
    
    #   functors
    conjunctions: list
    conjunction_frequency: int
    prepositions: list
    preposition_frequency: int
    determiners: list
    determiner_frequency: int
    predeterminers: list
    predeterminer_frequency: int
    pronouns: list
    pronoun_frequency: int
    
    #   extras
    tokens: list
    token_frequency: int
    lemmas: dict
    sluices: list
    sluice_frequency: int

@dataclass
class LexicalRichness():
    #   lexical variation
    ttr: int    #   type-token ratio
    mtld: int   #   measure of textual lexical diversity
    ndw: int    #  number of different words

    #   lexical density
    ld: int     #   lexical density


def measure_ttr(text: str) -> float:
    """
    Calculate the Type-Token Ratio for a given text sample.

    Args:
        text (str): the text to analyze using type-token ratio

    Returns:
        ttr (float): the calculated type-token ratio
    """

    words = text.lower().split(' ')
    unique_words = set(words)
    ttr = len(unique_words) / len(words)

    return ttr

def measure_ndw(words: list) -> float:
    """
    Calculates the NDW (Number of Different Words).

    Args:
        words (list): the input text to analyze.

    Returns:
        float: the NDW-ER50 score, which is the mean number of unique words across all samples.
    """

    unique_words = set(words)
    return len(unique_words)   

def measure_ndw_er50(words: list, num_samples=10, sample_size=50) -> float:
    """
    Calculates the NDW-ER50 (Number of Different Words - Estimated from Random 50-word samples).

    Args:
        words (list): the input text to analyze.
        num_samples (int): the number of random samples to take.
        sample_size (int): the size of each random word sample.

    Returns:
        float: the NDW-ER50 score, which is the mean number of unique words across all samples.
    """
    
    # Check if there are enough words to create samples
    if len(words) < sample_size:
        raise ValueError(f"Input text has fewer than {sample_size} words.")

    unique_word_counts = []
    
    for _ in range(num_samples):
        # Generate a random 50-word sample
        sample = random.sample(words, sample_size)
        
        # Count the number of unique words in the sample
        unique_word_count = len(Counter(sample))
        unique_word_counts.append(unique_word_count)
        
    # Calculate the mean of the unique word counts
    ndw_er50 = sum(unique_word_counts) / len(unique_word_counts)
    
    return ndw_er50

def measure_parts_of_speech():
    """
    
    Args:

    Returns:
    """

def measure_utterance_length():
    """
    
    Args:

    Returns:
    """

def measure_number_words(words: list) -> int:
    """
    Calculate the number of words.
    Args:
        words (list): the list of words to count

    Returns:
        int: the count of words
    """
    return len(words)

def measure_number_fillers():
    """
    
    Args:

    Returns:
    """

#   TODO: need to implement json/jsonl reader
def read_input(file: str) -> list:
    """
    Read a CSV or JSON/JSONL synthetic data file.

    Args:
        file: the csv or json/jsonl file to be read.

    Returns:
        synthetic_data: the 'transcript' column from a csv or the 'transcript' field from json/jsonl
    """
    synthetic_data = []
    base_path = os.path.join(os.getcwd(), 'data/')

    #   if filename contains csv
    if ".csv" in file:
        try:
            csv_file = base_path + file
            with open(csv_file, 'r', newline='') as f:
                csv_reader = csv.reader(f)
                for row in csv_reader:
                    synthetic_data.append(row[7]) # column 7 is 'transcript'
        except FileNotFoundError:
            print("Error: The csv file was not found.")
        except PermissionError:
            print("Error: Insufficient permissions to access the csv file.")
        except Exception as e:
            print(f"An unexpected error occurred: {e}")

    #   if filename contains json
    if ".json" in file or ".jsonl" in file:
        try:
            with open(file, 'r') as f:
                data = json.load(file)
        except FileNotFoundError:
            print("Error: The json file was not found.")
        except PermissionError:
            print("Error: Insufficient permissions to access the json file.")
        except Exception as e:
            print(f"An unexpected error occurred: {e}")

    return synthetic_data

#   TODO: need to design a data structure for 'transcripts'
def main(input: str, operation: str):
    synthetic_data = read_input(args.input)

    if args.operation == "ttr":
        ttrs = []
        for data in synthetic_data:
            #   TODO: should we tokenize first here?
            ttrs.append(measure_ttr(data.replace("Participant: ", "")))

    if args.operation == "ndw":
        ndws = []
        for data in synthetic_data:
            words = word_tokenize(data.replace("Participants: ", ""))
            if measure_number_words(words) > 50:
                ndws.append(measure_ndw_er50(words))
            else:
                ndws.append(measure_ndw(words))

    if args.operation == "count":
        word_counts = [] 
        for data in synthetic_data:
            words = word_tokenize(data.replace("Participants: ", ""))
            word_counts.append(measure_number_words(words))

        print(word_counts)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    
    requiredNamed = parser.add_argument_group('required named arguments')
    requiredNamed.add_argument("-in", "--input", required=True, help="specify the name of the transcript file")
    requiredNamed.add_argument("-op", "--operation", required=True, help="Valid operations types are: ttr, ndw-er50, count")

    args = parser.parse_args()
    main(args.input, args.operation)