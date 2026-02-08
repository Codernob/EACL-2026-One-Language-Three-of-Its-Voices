<img width="1903" height="394" alt="image" src="https://github.com/user-attachments/assets/51bf4866-bef8-4d0f-bb12-ac51c7aeaa21" />
<p align="center">
<img width="766" height="251" alt="image" src="https://github.com/user-attachments/assets/c7533a7b-1786-4216-9cf1-45f2e6516062" />
</p>

This repository contains code for my recently accepted paper "One Language, Three of Its Voices: Evaluating Multilingual LLMs Across Persian, Dari, and Tajiki on Translation and Understanding Tasks" accepted at the SilkRoadNLP workshop under EACL 2026.

# Datasets used

## Sentiment Analysis

- **SentiPers**  
  https://github.com/phosseini/SentiPers/tree/master

- **Digikala Comments Sentiment Analysis**  
  https://github.com/Arminkhayati/Digikala-comments-sentiment-analysis/tree/main

- **SnapFood Sentiment (HooshvareLab BERT-FA)**  
  https://huggingface.co/HooshvareLab/bert-fa-base-uncased-sentiment-snappfood


## Machine Translation

- **FLORES-200**
  - Repository: https://github.com/facebookresearch/flores/tree/main/flores200  
  - Dataset: https://tinyurl.com/flores200dataset

- **Tatoeba (OPUS)**  
  https://opus.nlpl.eu/Tatoeba/en&pes/v2023-04-12/Tatoeba


## Natural Language Inference

- **FarsTail**  
  https://github.com/dml-qom/FarsTail


## Question Answering

- **PQuAD**  
  https://github.com/AUT-NLP/PQuAD/tree/main

Download the datasets and keep them in their respective folders.

Run this command to install required dependencies. I recommend to create a new python environment. 

`pip install requirements.txt`

An NVIDIA GPU required for training.

and then run the script

`python scripts\run_experiments.py`

If you find my code useful in your work, do feel free to cite this repository.

Citation and paper pdf will be provided once the paper comes online.
