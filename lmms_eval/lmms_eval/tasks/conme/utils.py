import os
import json
import argparse
import pandas as pd
import string
import re
import nltk
import io
import ssl
import urllib.request

from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet
from tqdm import tqdm
from urllib.error import URLError
from getpass import getpass
from PIL import Image

from loguru import logger as eval_logger


def conme_doc_to_visual(doc):
    # try:
    #     return {f"{doc['image']}": [doc["image"].convert("RGB")]}
    # except:
    # print('\n Opening the image in a different way... \n Image is probably in bytes, string or different format. \n')
    # byts = ast.literal_eval(doc['image'])['bytes']
    images_folder = "/mnt/nushare2/data/mnulli/thesis/data/benchmarks/conme/"
    image_path = os.path.join(images_folder, doc["image"])

    image = Image.open(image_path).convert("RGB")
    return [image]


def conme_doc_to_text_mc(doc, lmms_eval_specific_kwargs=None):
    question = doc["question"]

    post = lmms_eval_specific_kwargs["post_prompt"]
    return question + "\n" + post


def conme_process_results(doc, results):
    """
    Processes string matching results and stores necessary information for metric computation.
    """
    pred = results[0]
    gt = doc["answer"]

    return {"string_matching_accuracy": {"prediction": pred, "answer": gt}}


def retrieve_special_characters(text):
    # First, look for a single letter between brackets,
    # allowing for optional spaces inside the parentheses.
    bracket_match = re.search(r"\(\s*([A-Ea-e])\s*\)", text)
    if bracket_match:
        # Return just the letter (without the brackets)
        return bracket_match.group(1)

    # If no bracket match, look for a standalone letter A–E.
    # This regex uses lookarounds to ensure the letter is not part of a larger word.
    standalone_match = re.search(r"(?<!\S)([A-Ea-e])(?!\S)", text)
    if standalone_match:
        return standalone_match.group(1)

    return None  # Return None if no match found.


def calculate_accuracy(results):
    # removing noise from outputs

    all_elements = len(results)
    pos = 0
    for result in results:
        pred = retrieve_special_characters(result["prediction"])
        ans = retrieve_special_characters(result["answer"])
        # print("")
        print("pred", result["prediction"], "pred after processing", pred, "ans", result["answer"], "ans after processing", ans)
        if pred is not None and ans is not None:
            if ans in pred:
                print("Yes")
                pos += 1

    accuracy = pos / all_elements

    return accuracy


def combine_accuracies(results):
    # Calculate accuracy for each source
    combined_accuracy = calculate_accuracy(results)

    eval_logger.info(f"conme accuracy:", combined_accuracy)

    return combined_accuracy
