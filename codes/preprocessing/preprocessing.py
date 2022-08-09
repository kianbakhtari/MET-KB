import os
import numpy as np
import nltk
import pandas as pd
import json
from tqdm import tqdm
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from autocorrect import Speller
from bs4 import BeautifulSoup
from nltk.corpus import stopwords
from nltk import word_tokenize
import string
import unidecode
import re

import warnings

warnings.filterwarnings('ignore')
# nltk.download('wordnet')
# nltk.download('punkt')
# nltk.download('omw-1.4')


class Preprocessor:
    def __init__(self):

        contraction_map_path = 'preprocessing/contraction_map.json'
        with open(contraction_map_path) as f:
            self.contraction_map = json.load(f)

#         nltk.download('stopwords')
        self.stoplist = stopwords.words('english')
        self.stoplist = set(self.stoplist).union({"'", "s", "S", "k", "."})

        self.w_tokenizer = nltk.tokenize.WhitespaceTokenizer()
        self.lemmatizer = nltk.stem.WordNetLemmatizer()

    def remove_newlines_tabs(self, text):
        formatted_text = text.replace('\\n', ' ').replace('\n', ' ').replace('\t', ' ')
        formatted_text = formatted_text.replace('\\', ' ').replace('. com', '.com')
        return formatted_text

    def remove_mentions_hashtagSigns(self, text, keep_hashtags=False):
        tokenized = text.split(" ")
        formatted_text = ""
        for w in tokenized:
            if not w:
                continue
            if w[0] == "@":
                continue
            if w[0] == "#":
                if not keep_hashtags:
                    continue
                formatted_text += f"{w[1:]} "
            else:
                formatted_text += f"{w} "

        return formatted_text.strip()

    def strip_html_tags(self, text):
        soup = BeautifulSoup(text, "html.parser")
        stripped_text = soup.get_text(separator=" ")
        return stripped_text

    def remove_punctuations(self, text):
        formatted = ""
        for char in text:
            if char in string.punctuation:
                continue
            else:
                formatted += char
        return formatted

    def remove_links(self, text):
        remove_https = re.sub(r'http\S+', '', text)
        remove_com = re.sub(r"\ [A-Za-z]*\.com", " ", remove_https)
        return remove_com

    def remove_whitespace(self, text):
        pattern = re.compile(r'\s+')
        Without_whitespace = re.sub(pattern, ' ', text)
        text = Without_whitespace.replace('?', ' ? ').replace(')', ') ')
        return text

    def accented_characters_removal(self, text):
        # Remove accented characters from text using unidecode.
        # Unidecode() - It takes unicode data & tries to represent it to ASCII characters. 
        text = unidecode.unidecode(text)
        return text

    def lower_casing_text(self, text):
        return text.lower()

    def reducing_incorrect_character_repeatation(self, text):
        Pattern_alpha = re.compile(r"([A-Za-z])\1{1,}", re.DOTALL)

        # Limiting all the  repeatation to two characters.
        Formatted_text = Pattern_alpha.sub(r"\1\1", text)

        # Pattern matching for all the punctuations that can occur
        Pattern_Punct = re.compile(r'([.,/#!$%^&*?;:{}=_`~()+-])\1{1,}')

        # Limiting punctuations in previously formatted string to only one.
        Combined_Formatted = Pattern_Punct.sub(r'\1', Formatted_text)

        # The below statement is replacing repeatation of spaces that occur more than two times with that of one occurrence.
        Final_Formatted = re.sub(' {2,}', ' ', Combined_Formatted)
        return Final_Formatted

    def expand_contractions(self, text):
        contraction_mapping = self.contraction_map
        list_Of_tokens = text.split(' ')
        for Word in list_Of_tokens:
            if Word in contraction_mapping:
                list_Of_tokens = [item.replace(Word, contraction_mapping[Word]) for item in list_Of_tokens]

        String_Of_tokens = ' '.join(str(e) for e in list_Of_tokens)
        return String_Of_tokens

    def removing_special_characters(self, text):
        formatted_text = re.sub(r"[^a-zA-Z0-9:$-,%.?!]+", ' ', text)
        return formatted_text

    def remove_numbers(self, text):
        formatted_text = re.sub(r"[^a-zA-Z:$-,%.?!]+", ' ', text)
        return formatted_text

    def removing_stopwords(self, text):
        No_StopWords = [word for word in word_tokenize(text) if word.lower() not in self.stoplist]
        words_string = ' '.join(No_StopWords)
        return words_string

    def spelling_correction(self, text):
        spell = Speller(lang='en')
        Corrected_text = spell(text)
        return Corrected_text

    def cleaning_and_preprocessing(self, text,
                                   remove_newlines_=True,
                                   remove_mentions_hashtagSigns_=True,
                                   strip_html_tags_=True,
                                   remove_punctuations_=True,
                                   remove_links_=True,
                                   remove_whitespace_=True,
                                   accented_characters_removal_=True,
                                   lower_casing_text_=True,
                                   reducing_incorrect_character_repeatation_=True,
                                   expand_contractions_=True,
                                   remove_numbers_=True,
                                   removing_stopwords_=True,
                                   spelling_correction_=False):

        """
        input: a single text (e.g., a text or a sentence with type string).
        output: a single text (e.g., a text or a sentence with type string) which is clean:)
        """

        ## Cleaning
     
        if remove_newlines_:
            text = self.remove_newlines_tabs(text)
        if remove_mentions_hashtagSigns_:
            text = self.remove_mentions_hashtagSigns(text)
        if strip_html_tags_:
            text = self.strip_html_tags(text)
        if remove_punctuations_:
            text = self.remove_punctuations(text)
        if remove_links_:
            text = self.remove_links(text)
        if remove_whitespace_:
            text = self.remove_whitespace(text)
        if accented_characters_removal_:
            text = self.accented_characters_removal(text)
        if lower_casing_text_:
            text = self.lower_casing_text(text)
        if reducing_incorrect_character_repeatation_:
            text = self.reducing_incorrect_character_repeatation(text)
        if expand_contractions_:
            text = self.expand_contractions(text)
        if remove_numbers_:
            text = self.remove_numbers(text)

        ## preprocessing
        if removing_stopwords_:
            text = self.removing_stopwords(text)
        if spelling_correction_:
            text = self.spelling_correction(text)
        return text

    def clean_data(self, df, col):
        """
        gets whole df as input.
        returns df with new column called cleaned texts.
        """
        cleaned_texts = []
        for text in tqdm(df[col]):
            cleaned_text = self.cleaning_and_preprocessing(text)
            cleaned_texts.append(cleaned_text)

        df[f"cleaned_{col}"] = cleaned_texts
        return df

    def lemmatize_data(self, df, col):
        """
        gets whole df as input.
        returns df with new column called lemmatized texts.
        """
        lemmatized_texts = []
        for text in tqdm(df[f"cleaned_{col}"]):
            lemma = [self.lemmatizer.lemmatize(w, 'v') for w in self.w_tokenizer.tokenize(text)]
            lemma = ' '.join(lemma)
            lemmatized_texts.append(lemma)

        df[f"lemmatized_{col}"] = lemmatized_texts
        return df

    def tokenize_data(self, df, col):
        pattern = r'''(?x)          # set flag to allow verbose regexps
                (?:[A-Z]\.)+        # abbreviations, e.g. U.S.A.
              | \w+(?:-\w+)*        # words with optional internal hyphens
              | \$?\d+(?:\.\d+)?%?\s?  # currency and percentages, e.g. $12.40, 82%
              | \.\.\.              # ellipsis
              | [][.,;"'?():_`-]    # these are separate tokens; includes ], [
            '''

        pattern = re.compile(pattern)
        nltk_tokenized_texts = [nltk.regexp_tokenize(text, pattern) for text in df[f"lemmatized_{col}"]]

        nltk_tokenized_texts_final = []
        for text_tokens in nltk_tokenized_texts:
            text_tokens_final = []
            for word in text_tokens:
                if word not in self.stoplist:
                    text_tokens_final.append(word)
            nltk_tokenized_texts_final.append(text_tokens_final)

        df[f'tokens_{col}'] = nltk_tokenized_texts_final
        return df

    def perform_clean_lemmatize_tokenize(self, df, col: str):
        df = self.clean_data(df, col)
        df = self.lemmatize_data(df, col)
        df = self.tokenize_data(df, col)
        return df

    def clean_query(self, text):
        text = self.cleaning_and_preprocessing(text)
        lemma = [self.lemmatizer.lemmatize(w, 'v') for w in self.w_tokenizer.tokenize(text)]
        text = ' '.join(lemma)
        pattern = r'''(?x)          # set flag to allow verbose regexps
                (?:[A-Z]\.)+        # abbreviations, e.g. U.S.A.
              | \w+(?:-\w+)*        # words with optional internal hyphens
              | \$?\d+(?:\.\d+)?%?\s?  # currency and percentages, e.g. $12.40, 82%
              | \.\.\.              # ellipsis
              | [][.,;"'?():_`-]    # these are separate tokens; includes ], [
            '''

        pattern = re.compile(pattern)
        return nltk.regexp_tokenize(text, pattern)
    