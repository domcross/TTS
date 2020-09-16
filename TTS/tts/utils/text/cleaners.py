'''
Cleaners are transformations that run over the input text at both training and eval time.

Cleaners can be selected by passing a comma-delimited list of cleaner names as the "cleaners"
hyperparameter. Some cleaners are English-specific. You'll typically want to use:
  1. "english_cleaners" for English text
  2. "transliteration_cleaners" for non-English text that can be transliterated to ASCII using
     the Unidecode library (https://pypi.python.org/pypi/Unidecode)
  3. "basic_cleaners" if you do not want to transliterate (in this case, you should also update
     the symbols in symbols.py to match your data).
'''

import pkg_resources
installed = {pkg.key for pkg in pkg_resources.working_set}  #pylint: disable=not-an-iterable
import re
if 'german_transliterate' in installed:
    from german_transliterate.core import GermanTransliterate  # https://github.com/repodiac/german_transliterate
if 'phonemizer' in installed:
    from phonemizer.phonemize import phonemize
from unidecode import unidecode
from .number_norm import normalize_numbers

# Regular expression matching whitespace:
_whitespace_re = re.compile(r'\s+')

# List of (regular expression, replacement) pairs for abbreviations:
_abbreviations = [(re.compile('\\b%s\\.' % x[0], re.IGNORECASE), x[1])
                  for x in [
                      ('mrs', 'misess'),
                      ('mr', 'mister'),
                      ('dr', 'doctor'),
                      ('st', 'saint'),
                      ('co', 'company'),
                      ('jr', 'junior'),
                      ('maj', 'major'),
                      ('gen', 'general'),
                      ('drs', 'doctors'),
                      ('rev', 'reverend'),
                      ('lt', 'lieutenant'),
                      ('hon', 'honorable'),
                      ('sgt', 'sergeant'),
                      ('capt', 'captain'),
                      ('esq', 'esquire'),
                      ('ltd', 'limited'),
                      ('col', 'colonel'),
                      ('ft', 'fort'),
                  ]]


def expand_abbreviations(text):
    for regex, replacement in _abbreviations:
        text = re.sub(regex, replacement, text)
    return text


def expand_numbers(text):
    return normalize_numbers(text)


def lowercase(text):
    return text.lower()


def collapse_whitespace(text):
    return re.sub(_whitespace_re, ' ', text).strip()


def convert_to_ascii(text):
    return unidecode(text)


def remove_aux_symbols(text):
    text = re.sub(r'[\<\>\(\)\[\]\"]+', '', text)
    return text

def replace_symbols(text, lang='en'):
    text = text.replace(';', ',')
    text = text.replace('-', ' ')
    text = text.replace(':', ' ')
    if lang == 'en':
        text = text.replace('&', 'and')
    elif lang == 'pt':
        text = text.replace('&', ' e ')
    return text

def basic_cleaners(text):
    '''Basic pipeline that lowercases and collapses whitespace without transliteration.'''
    text = lowercase(text)
    text = collapse_whitespace(text)
    return text


def transliteration_cleaners(text):
    '''Pipeline for non-English text that transliterates to ASCII.'''
    text = convert_to_ascii(text)
    text = lowercase(text)
    text = collapse_whitespace(text)
    return text


def basic_german_cleaners(text):
    '''Pipeline for German text'''
    text = lowercase(text)
    text = collapse_whitespace(text)
    return text


# TODO: elaborate it
def basic_turkish_cleaners(text):
    '''Pipeline for Turkish text'''
    text = text.replace("I", "ı")
    text = lowercase(text)
    text = collapse_whitespace(text)
    return text

def english_cleaners(text):
    '''Pipeline for English text, including number and abbreviation expansion.'''
    text = convert_to_ascii(text)
    text = lowercase(text)
    text = expand_numbers(text)
    text = expand_abbreviations(text)
    text = replace_symbols(text)
    text = remove_aux_symbols(text)
    text = collapse_whitespace(text)
    return text

def portuguese_cleaners(text):
    '''Basic pipeline for Portuguese text. There is no need to expand abbreviation and
        numbers, phonemizer already does that'''
    text = lowercase(text)
    text = replace_symbols(text, lang='pt')
    text = remove_aux_symbols(text)
    text = collapse_whitespace(text)
    return text

def phoneme_cleaners(text):
    '''Pipeline for phonemes mode, including number and abbreviation expansion.'''
    text = convert_to_ascii(text)
    text = expand_numbers(text)
    text = expand_abbreviations(text)
    text = replace_symbols(text)
    text = remove_aux_symbols(text)
    text = collapse_whitespace(text)
    return text

def german_phoneme_cleaners(text):
    if 'german_transliterate' in installed:
        return GermanTransliterate(replace={';': ',', ':': ' '}, sep_abbreviation=' -- ').transliterate(text)
    elif 'phonemizer' in installed and 'espeakng' in installed:
        text = convert_to_ascii(text)
        #text = expand_numbers(text)
        #text = expand_abbreviations(text)
        text = replace_symbols(text)
        text = remove_aux_symbols(text)
        text = collapse_whitespace(text)
        return phonemize(text, language='de', backend='espeak'))
    else:
        raise NotImplementedError("german_phoneme_cleaners requires package 'german_transliterate' or package 'phonemizer' and 'espeakng' installed!")
