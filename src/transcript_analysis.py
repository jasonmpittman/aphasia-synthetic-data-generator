__author__ = "Jason M. Pittman"
__copyright__ = "Copyright 2025"
__credits__ = ["Jason M. Pittman"]
__license__ = "Apache License 2.0"
__version__ = "0.1.0"
__maintainer__ = "Jason M. Pittman"
__status__ = "Research"

"""



"""

import argparse
import json
from dataclasses import dataclass

import nltk
nltk.download()

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
class LanguageMeasures():
    #   lexical diversity
    ttr: int
    mtld: int


def measure_lexcical_diversity(text: str):
    """
    
    Args:

    Returns:
    """


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

def read_json(file: str) -> list:
    """
    
    Args:

    Returns:
    """
    synthetic_data = []

    try:
        with open(file, 'r') as f:
            data = json.load(file)
    except FileNotFoundError:
        print("Error: The file was not found.")
    except PermissionError:
        print("Error: Insufficient permissions to access the file.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

    #   if file contains csv

    #   if file contains json

    return synthetic_data

def main(input: str, operation: str):
    
    if args.operation == "":
        pass


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    
    requiredNamed = parser.add_argument_group('required named arguments')
    requiredNamed.add_argument("-in", "--input", required=True, help="specify the name of the transcript file")
    requiredNamed.add_argument("-op", "--operation", required=True, help="Valid operations types are: ")

    args = parser.parse_args()
    main(args.input, args.operation)