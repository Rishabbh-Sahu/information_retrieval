# -*- coding: utf-8 -*-
"""
@author: rishabbh-sahu
"""
import string
import re     #for text pre-processing


def text_preprocessing(text_str: str, remove_digit=True, remove_punctuations=True):
    """
    text pre-preprocess of the input like normalization, special tags (html,chars) removal, removing digits is optional
    text_str: string to be pre-processed
    return: cleaned text
    """
    txt = ''
    # normalize to lower case
    txt = text_str.lower()
    
    # remove tags
    txt = re.sub('</?.*?>', ' <> ', txt)

    # remove special characters
    txt = re.sub(r'\W+', ' ', txt).strip()

    if remove_digit:
        # remove digits
        txt = re.sub(r'\d+', '', txt).strip()

    if remove_punctuations:
        # remove punctuations from string package
        txt = "".join([char for char in txt if char not in string.punctuation])

    return txt
