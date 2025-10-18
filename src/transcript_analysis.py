__author__ = "Jason M. Pittman"
__copyright__ = "Copyright 2025"
__credits__ = ["Jason M. Pittman"]
__license__ = "Apache License 2.0"
__version__ = "0.1.0"
__maintainer__ = "Jason M. Pittman"
__status__ = "Research"

"""



"""

import os
import re
import csv
import json
import argparse
from dataclasses import dataclass

import nltk
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


def measure_ttr(text: str):
    """
    Calculate the Type-Token Ratio for a given text sample.

    Args:
        text: the text to analyze using type-token ratio

    Returns:
        ttr: the calculated type-token ratio
    """

    words = text.lower().split(' ')
    unique_words = set(words)
    ttr = len(unique_words) / len(words)

    return ttr


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

def measure_number_words():
    """
    
    Args:

    Returns:
    """

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

def main(input: str, operation: str):
    synthetic_data = read_input(args.input)

    if args.operation == "ttr":
        ttrs = []
        for data in synthetic_data:
            ttrs.append(measure_ttr(data.replace("Participant: ", "")))
        

        print(ttrs)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    
    requiredNamed = parser.add_argument_group('required named arguments')
    requiredNamed.add_argument("-in", "--input", required=True, help="specify the name of the transcript file")
    requiredNamed.add_argument("-op", "--operation", required=True, help="Valid operations types are: ")

    args = parser.parse_args()
    main(args.input, args.operation)