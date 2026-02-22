# Aphasia Synthetic Data Generators

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.17542988.svg)](https://doi.org/10.5281/zenodo.17542988)

Date created: October 07, 2025  
Last updated: 0ctober 19, 2025

# Project Description
This project provides source code and instructions associated with two programmatic methods of generating synthetic transcripts (see [Instructions](#instructions)). The synthetic transcripts emulate the Cat Rescue aphasia clinical diagnostic test. Details of each method are described in the associated paper (see [Links](#links)).

# People
**Creator** - Jason M. Pittman<sup>1</sup>  
**Collaborators** - Yesenia Medina-Santos<sup>2</sup> | Anton Phillips Jr<sup>2</sup> | Brie Stark<sup>2</sup>

# Institute
<sup>1</sup> University of Maryland Global Campus USA  
<sup>2</sup> Indiana University Bloomington, Department of Speech, Language and Hearing Sciences USA

# Project Architecture
```
.
├── README.md
├── data
├── docs
│   ├── article
│   ├── notes
│   └── presentation
├── src
```

# Instructions
The project was developed using Python 3.9.6 on an Apple M4 silicon system.

To install the necessary packages, run:
``` 
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

**Note:** If you are on a different operating system, or wish to use a different project directory structure, please edit paths in the generator and analyses programs.

## Procedural Method
The *procedural* method can be run using the following:
```
python3 src/procedural_generatory.py
```

This will:
- Produce 10,000 examples evenly split across severities (Mild/Moderate/Severe/Very Severe).
- Uses sentence-level templates of the cat-rescue scenario.
- Applies severity-specific augmentations (dropping, paraphasias, fillers).
- Computes Word and CIU metrics using simplified lexical rules.
- Writes three JSONL files: train.jsonl, validation.jsonl, and test.jsonl.

Output is written to the `data/procedural` directory by default.

## ML (LLM) Method
The *large language model* method can be run using a command structure concisting of calling the program followed by 5 arguments. For example, the specific prompts used to generate the transcripts in the [data](data/llm/) directory were:
```
python3 src/llm_generator.py \
--prompt-pack-dir src/cat_rescue_synthetic_promptpack \
--output-dir data/llm \
--prompt-model llama3.1-8b \
--hf-model-id meta-llama/Meta-Llama-3.1-8B-Instruct \
--samples-per-severity 8

python3 src/llm_generator.py \
--prompt-pack-dir src/cat_rescue_synthetic_promptpack \
--output-dir data/llm \
--prompt-model mistral \
--hf-model-id mistralai/Mistral-7B-Instruct-v0.3 \
--samples-per-severity 8
```

Output is written to the `data\llm` directory by default.

## Analysis
If you want to perform analysis of your generated transcripts, you can run:
```
python3 src/transcript_analysis.py -in <file> -s <severity> -op <operation>
```

If you want the additional Word and CIU analyses for the ML/LLM method, you can run:
```
python3 src/augment_llm_metrics.py
```

Word and CIU analysis for the procedural method can be performed manually using the columns or keys in the output files.


# Links
(Paper)[https://arxiv.org/abs/2510.24817]


