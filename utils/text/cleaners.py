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

import re
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


def replace_symbols(text):
    text = text.replace(';', ',')
    text = text.replace('-', ' ')
    text = text.replace(':', ' ')
    text = text.replace('&', 'and')
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
    return GermanCleaner().clean(text)


def german_minimal_cleaners(text):
    return GermanCleaner(replace={'-­': ' '}, sep_abbreviation=' ').clean(text)


class GermanCleaner:
    import re

    SEP_ABBR_MASK = '\x00'  # NULL byte for MASKING separator

    # units
    UNIT_0 = {
        'mg': 'milligramm',
        'kg': 'kilogramm',
        'g': 'gramm',
        'nm': 'nanometer',
        'µm': 'mikrometer',
        'mm': 'millimeter',
        'mm^2': 'quadratmillimeter',
        'mm²': 'quadratmillimeter',
        'cm': 'zentimeter',
        'cm^2': 'quadratzentimeter',
        'cm²': 'quadratzentimeter',
        'cm^3': 'kubikzentimeter',
        'cm³': 'kubikzentimeter',
        'dm': 'dezimeter',
        'm': 'meter',
        'm^2': 'quadratmeter',
        'm²': 'quadratmeter',
        'm^3': 'kubikmeter',
        'm³': 'kubikmeter',
        'km': 'kilometer',
        'km^2': 'quadratkilometer',
        'km²': 'quadratkilometer',
        'ha': 'hektar',
        'w': 'watt',
        'j': 'joule',
        'kj': 'kilojoule',
        'k_b': 'kilobyte',
        'm_b': 'megabyte',
        'g_b': 'gigabyte',
        't_b': 'terabyte',
        'p_b': 'petabyte',
        'k_w': 'kilowatt',
        'kb': 'kilobyte',
        'mb': 'megabyte',
        'gb': 'gigabyte',
        'tb': 'terabyte',
        'pb': 'petabyte',
        'kw': 'kilowatt',
        'm_w': 'megawatt',
        'g_w': 'gigawatt',
        'mw': 'megawatt',
        'gw': 'gigawatt',
    }

    UNIT_1 = {
        't': 'tonnen',
        'kt': 'kilotonnen',
        'mt': 'megatonnen',
        'kwh': 'kilowattstunden',
        'mwh': 'megawattstunden',
        'gwh': 'gigawattstunden',
        'kal': 'kalorien',
        'cal': 'kalorien',
        'mia': 'milliarden',
        'mrd': 'milliarden',
        'md': 'milliarden',
        'brd': 'billiarden',
        'ns': 'nanosekunden',
        'µs': 'mikrosekunden',
        'ms': 'millisekunden',
        's': 'sekunden',
        'sek': 'sekunden',
        'm': 'minuten',
        'min': 'minuten',
        'h': 'stunden',
    }

    UNIT_2 = {
        'm': 'millionen',
        'mio': 'millionen',
        'mill': 'millionen',
        'bill': 'billionen',
    }

    ABBREVIATION = {
        'fr': 'frau',
        'hr': 'herr',
        'dr': 'doktor',
        'prof': 'professor',
        'jprof': 'juniorprofessor',
        'jun.prof': 'juniorprofessor',
        'mag': 'magister',
        'bsc': 'bachelor of science',
        'msc': 'master of science',
        'st': 'sankt',
        'skt': 'sankt',
    }

    # dates
    TIME = {
        'ns': 'nanosekunden',
        'µs': 'mikrosekunden',
        'ms': 'millisekunden',
        's': 'sekunden',
        'sek': 'sekunden',
        'm': 'minuten',
        'min': 'minuten',
        'h': 'stunden',
    }

    NUMBER_MONTH = {
        '1': 'januar',
        '01': 'januar',
        '2': 'februar',
        '02': 'februar',
        '3': 'märz',
        '03': 'märz',
        '4': 'april',
        '04': 'april',
        '5': 'mai',
        '05': 'mai',
        '6': 'juni',
        '06': 'juni',
        '7': 'juli',
        '07': 'juli',
        '8': 'august',
        '08': 'august',
        '9': 'september',
        '09': 'september',
        '10': 'oktober',
        '11': 'november',
        '12': 'dezember',
    }

    WEEKDAY = {
        'mo': 'montag',
        'di': 'dienstag',
        'mi': 'mittwoch',
        'do': 'donnerstag',
        'fr': 'freitag',
        'sa': 'samstag',
        'so': 'sonntag',
    }

    ABBREVIATION_MONTH = {
        'januar': 'januar',
        'jänner': 'jänner',
        'februar': 'februar',
        'märz': 'märz',
        'april': 'april',
        'mai': 'mai',
        'juni': 'juni',
        'juli': 'juli',
        'august': 'august',
        'september': 'september',
        'oktober': 'oktober',
        'november': 'november',
        'dezember': 'dezember',
        'jan': 'januar',
        'jän': 'jänner',
        'feb': 'februar',
        'mrz': 'märz',
        'mär': 'märz',
        'apr': 'april',
        'jun': 'juni',
        'jul': 'juli',
        'aug': 'august',
        'sep': 'september',
        'okt': 'oktober',
        'nov': 'november',
        'dez': 'dezember',
    }

    SPECIAL_TRANSLITERATE = {
        '.*\+/\-.*': ('+/-', 'plus minus'),
        '.*&.*': ('&', ' und '),
        '(^|(?<=[\.!?;:\-\s]))([a-z]{0,1}|[\d]+)\s{0,1}[^-]\-\s{0,1}([\d]+|[a-z]{0,1})($|(?=[\.!?;:\-\s]+))': (
            '-', ' bis '),
        # TODO: include more dash operators in unicode...
        '\\b[02-9]+\s{0,1}/\s{0,1}\d+\\b': ('/', ' von '),
        '\\b1/10\\b': ('1/10', 'ein zehntel'),
        '\\b⅒\\b': ('⅒', 'ein zehntel'),
        '\\b1/9\\b': ('1/9', 'ein neuntel'),
        '\\b⅑\\b': ('⅑', 'ein neuntel'),
        '\\b1/8\\b': ('1/8', 'ein achtel'),
        '\\b⅛\\b': ('⅛', 'ein achtel'),
        '\\b1/7\\b': ('1/7', 'ein siebtel'),
        '\\b⅐\\b': ('⅐', 'ein siebtel'),
        '\\b1/6\\b': ('1/6', 'ein sechstel'),
        '\\b⅙\\b': ('⅙', 'ein sechstel'),
        '\\b1/5\\b': ('1/5', 'ein fünftel'),
        '\\b⅕\\b': ('⅕', 'ein fünftel'),
        '\\b1/4\\b': ('1/4', 'ein viertel'),
        '\\b¼\\b': ('¼', 'ein viertel'),
        '\\b1/3\\b': ('1/3', 'ein drittel'),
        '\\b⅓\\b': ('⅓', 'ein drittel'),
        '\\b1/2\\b': ('1/2', 'ein halb'),
        '\\b½\\b': ('½', 'ein halb'),
        '\\b1000\\b': ('1000', 'tausend'),
    }

    # order is important: longer strings before substrings, e.g. "<=" before "=" and "<"
    MATH_SYMBOL_TRANSLITERATE = {
        '+': ' plus ',
        '-': ' minus ',
        '−': ' minus ',
        '/': ' geteilt durch ',
        '\\': ' modulo ',
        '*': ' mal ',
        '×': ' mal ',
        'x': ' mal ',
        '>=': ' größer gleich ',
        '≥': ' größer gleich ',
        '<=': ' größer gleich ',
        '≤': ' größer gleich ',
        '==': ' äquvivalent zu ',
        '=': ' gleich ',
        '≍': ' äquvivalent zu ',
        '>': ' größer ',
        '<': ' kleiner ',
    }

    AUX_SYMBOL_TRANSLITERATE = {
        ('(', ')'): '_in klammern_',
        ('[', ']'): '_in klammern_',
        ('"', '"'): '_in anführungszeichen_',
        ('\'', '\''): '_zitat_',
    }

    # currencies
    CURRENCY_SYMBOL = {
        'g_b_p': 'britische pfund',  # '_' is used to mask separator used with abbreviations (see sep_abbreviation=...)
        '£': 'pfund',
        '$': 'dollar',
        'e_u_r': 'euro',
        't_e_u_r_o': 'tausend euro',
        't_e_u_r': 'tausend euro',
        # 'teuro': 'tausend euro',
        't€': 'tausend euro',
        '€': 'euro',
        't_d_m': 'tausend d-mark',
        'd_m': 'd-mark',
        'd_k_k': 'dänische kronen',
        's_e_k': 'schwedische kronen',
    }

    CURRENCY_MAGNITUDE = ['\\bmia\\b', '\\bmia\.\\b', '\\bmrd\\b', '\\bmrd\.\\b', '\\bmd\\b', '\\bmd\.\\b',
                          '\\bmilliarde[n]{0,1}\\b', '\\bbrd\\b', '\\bbrd\.\\b', '\\bbilliarde[n]{0,1}\\b',
                          '\\bmio\\b', '\\bmio\.\\b', '\\bmill\\b', '\\bmill\.\\b', '\\bmillion(en){0,1}\\b',
                          '\\bbill\\b', '\\bbill\.\\b', '\\bbillion(en){0,1}\\b', '\\btausend\\b']

    # source for "unicode-to-ascii" mappings (adapted):
    # https://github.com/AASHISHAG/deepspeech-german/blob/master/pre-processing/text_cleaning.py
    UNICODE_TO_ASCII = {
        'àáâãåāăąǟǡǻȁȃȧ': 'a',
        'æǣǽ': 'ä',
        'çćĉċč': 'c',
        'ďđ': 'd',
        'èéêëēĕėęěȅȇȩε': 'e',
        'ĝğġģǥǧǵ': 'g',
        'ĥħȟ': 'h',
        'ìíîïĩīĭįıȉȋ': 'i',
        'ĵǰ': 'j',
        'ķĸǩǩκ': 'k',
        'ĺļľŀł': 'l',
        'м': 'm',
        'ñńņňŉŋǹ': 'n',
        'òóôõōŏőǫǭȍȏðο': 'o',
        'œøǿ': 'ö',
        'ŕŗřȑȓ': 'r',
        'śŝşšș': 's',
        'ţťŧț': 't',
        'ùúûũūŭůűųȕȗ': 'u',
        'ŵ': 'w',
        'ýÿŷ': 'y',
        'źżžȥ': 'z',
    }

    REX_DETECT_ABBREVIATION = re.compile('(^|(?<=[\.!?;:\-\s]))([A-Z]{2,}|([A-Z]\.){2,})($|(?=[\.!?;:\-\s]+))')
    REX_DETECT_WEEKDAY = re.compile('\\b(' + '|'.join(WEEKDAY.keys()) + ')\\b')
    REX_DETECT_MONTH = re.compile('\\b(' +  '|'.join(ABBREVIATION_MONTH.keys()) + ')\\b')
    REX_DETECT_TIME_OF_DAY = re.compile('(\\b(([0-1][0-9]|2[0-3])[\.\:]|[0-9][\.\:])([0-5][0-9]|[0-9])\s{0,1}(h|uhr){0,1}\\b)')
    REX_DETECT_TIMESTAMP = re.compile('(\\b\d+(h|std){0,1}:([0-5][0-9]|[0-9])(m|min){0,1}(:([0-5][0-9]|[0-9])(s|sek|sec){0,1}){0,1}\\b)')
    REX_DETECT_DATE = re.compile(
        '(([1-9]|(0(1|2|3|4|5|6|7|8|9))|(1[0-9])|(2[0-9])|30|31)\.(((([1-9]|0(1|2|3|4|5|6|7|8|9))|(10|11|12))\.)|(\s{0,1}('
        +'|'.join(ABBREVIATION_MONTH.keys())+')(\.|\\b)))(\s{0,1}\d\d\d\d|\s{0,1}\d\d){0,1})')
    REX_DETECT_ORDINAL = re.compile('[\(\[]{0,1}\d+\.[\)\]]{0,1}')
    REX_DETECT_NUMBER = re.compile('([\+\-]{0,1}\d+[\d\.,]*)')
    REX_DETECT_WHITESPACE_SEQ = re.compile('\s+')

    def __init__(self,
                 transliterate=['acronym', 'accent_peculiarity', 'amount_money', 'date', 'timestamp',
                                'time_of_day', 'ordinal', 'special', 'math_symbol', 'aux_symbol'],
                                #'weekday', 'month',
                 replace={';': ',', ':': ' '},
                 sep_abbreviation=' -- '
                 ):
        self.transliterate = transliterate
        self.replace = replace
        self.sep_abbreviation = sep_abbreviation

        escaped_cursym = [re.escape(it) for it in self.CURRENCY_SYMBOL.keys()]
        self.rstring_cursym_escaped = '|'.join(escaped_cursym).replace('_', self.SEP_ABBR_MASK)
        self.rstring_curmagn = '|'.join(self.CURRENCY_MAGNITUDE)
        self.REX_DETECT_CURRENCY_SYMBOL = re.compile(self.rstring_cursym_escaped)
        self.REX_DETECT_CURRENCY_MAGNITUDE = re.compile(self.rstring_curmagn)

        self.REX_DETECT_CURRENCY = re.compile( \
            '(^|(?<=[\.!?;:\-\(\)\[\]\s]))(([\+\-]{0,1}\d+[\d\.,]*\s*('
            + self.rstring_curmagn + '){0,1}\s*' \
            '(' + self.rstring_cursym_escaped + '))|' \
            '((' + self.rstring_cursym_escaped + ')\s*[\+\-]{0,1}\d+[\d\.,]*\s*(' + self.rstring_curmagn + '){0,1})|' \
            '(' + self.rstring_cursym_escaped + '))($|(?=[\.!?;:\-\(\)\[\]\s]+))')

    def clean(self, text):
        from num2words import num2words

        if 'acronym' in self.transliterate:
            # TRANSLITERATE ABBREVIATONS (WITH ONLY CAPITAL LETTERS)
            abbr_expanded = []
            for abbr in self.REX_DETECT_ABBREVIATION.finditer(text):
                abbr_expanded.append(
                    (abbr.group(0), self.SEP_ABBR_MASK.join([c for c in abbr.group(0).replace('.', '')])))
            for m in abbr_expanded:
                text = text.replace(m[0], m[1], 1)

        # MAKE LOWERCASE
        text = text.lower()

        # REPLACE ACCENT "PECULIARITIES" WITH ASCII counterparts
        if 'accent_peculiarity' in self.transliterate:
            for chars, mapped in self.UNICODE_TO_ASCII.items():
                text = re.sub('|'.join([c for c in chars]), mapped, text)

        # TRANSLITERATE CURRENCIES
        if 'amount_money' in self.transliterate:
            diff_len = 0
            for mc in self.REX_DETECT_CURRENCY.finditer(text):
                m_symbol = self.REX_DETECT_CURRENCY_SYMBOL.search(mc.group(0))
                m_magnitude = self.REX_DETECT_CURRENCY_MAGNITUDE.search(mc.group(0))
                m_number = self.REX_DETECT_NUMBER.search(mc.group(0))

                number = m_number.group(0) if m_number else ''
                if not m_magnitude and ',' in number:
                    number = number.replace(',', ' ' + self.CURRENCY_SYMBOL[
                        m_symbol.group(0).replace(self.SEP_ABBR_MASK, '_')] + ' ')
                    rearranged_currency_term = number
                else:
                    rearranged_currency_term = number + ' ' if m_number else ''
                    rearranged_currency_term += m_magnitude.group(0) + ' ' if m_magnitude else ''
                    rearranged_currency_term += self.CURRENCY_SYMBOL[m_symbol.group(0).replace(self.SEP_ABBR_MASK, '_')]
                text = text[:mc.start() + diff_len] + rearranged_currency_term + text[mc.end() + diff_len:]
                diff_len = len(rearranged_currency_term) - (mc.end() - (mc.start() + diff_len))

        # TRANSLITERATE TIMESTAMP or DATE
        # OPTIONAL, TODO: cover also time durations
        if 'date' in self.transliterate:
            for date_m in self.REX_DETECT_DATE.finditer(text):
                frags = date_m.group(0).split('.')
                if ' ' in frags[-1]:
                    space_split = frags[-1].strip().split(' ')
                    del(frags[-1])
                    frags.extend(space_split)
                day = num2words(frags[0], lang='de', to='ordinal')
                if date_m.start() > 1 and text[date_m.start()-2:date_m.start()] in ('m ', 'n '):
                    day += 'n'
                if frags[1].strip() in self.ABBREVIATION_MONTH:
                    month = self.ABBREVIATION_MONTH[frags[1].strip()]
                else:
                    month = self.NUMBER_MONTH[frags[1].strip()]
                year = ''
                if len(frags) == 3 and frags[2]:
                    year = num2words(frags[2], lang='de', to='year')
                text = self.REX_DETECT_DATE.sub(day + ' ' + month + (' ' + year if year else ''), text, count=1)

        if 'timestamp' in self.transliterate:
            for timestamp_m in self.REX_DETECT_TIMESTAMP.finditer(text):
                ts = timestamp_m.group(0)
                ts_split = ts.split(':')
                if len(ts_split) == 2:
                    if int(ts_split[0].replace('h','').replace('std', '')) == 1:
                        ts = 'eine stunde '
                    else:
                        ts = ts_split[0] + ' stunden '
                    if int(ts_split[1].replace('m','').replace('min', '')) == 1:
                        ts += 'eine minute'
                    else:
                        ts += ts_split[1] + ' minuten'
                # assume len=3
                else:
                    if int(ts_split[0].replace('h','').replace('std', '')) == 1:
                        ts = 'eine stunde '
                    else:
                        ts = ts_split[0] + ' stunden '
                    if int(ts_split[1].replace('min', '').replace('m','')) == 1:
                        ts += 'eine minute '
                    else:
                        ts += ts_split[1].replace('min', '').replace('m','') + ' minuten '
                    if int(ts_split[2].replace('sek', '').replace('sec', '').replace('s','')) == 1:
                        ts += 'eine sekunde'
                    else:
                        ts += ts_split[2].replace('sek', '').replace('sec', '').replace('s','') + ' sekunden'

                text = text[:timestamp_m.start()] + ts + text[timestamp_m.end():]

        if 'time_of_day' in self.transliterate:
            # TODO: cover also other ways of transliterating time of day (e.g. 07.45h as "dreiviertel acht")
            for time_m in self.REX_DETECT_TIME_OF_DAY.finditer(text):
                tod = text[time_m.start():time_m.end()].replace('uhr', '').replace('h', '').replace(':',' uhr ').replace('.',' uhr ')
                if int(tod.split(' uhr ')[0]) == 1:
                    tod = 'ein uhr ' + tod.split(' uhr ')[1]
                text = text[:time_m.start()] + tod + text[time_m.end():]

        # ITERATE OVER SINGLE WORDS (split by space character)
        cleaned = []
        split_text = text.split(' ')
        idx = 0
        for word in split_text:
            if not word:
                continue

            # EXPAND CHARACTERS AND TERMS for transliteration
            for tr in self.transliterate:
                # WEEKDAYS or MONTHs
                if tr == 'weekday':
                    if self.REX_DETECT_WEEKDAY.match(word):
                        word = self.WEEKDAY[word.replace('.','')] # TODO: replace every aux char!

                elif tr == 'month':
                    if self.REX_DETECT_MONTH.match(word):
                        word = self.ABBREVIATION_MONTH[word.replace('.', '')]

                # ORDINAL NUMBERS
                elif tr == 'ordinal':
                    if self.REX_DETECT_ORDINAL.match(word) and word.endswith('.'):
                        if idx < (len(split_text) - 1) and split_text[idx + 1] not in self.CURRENCY_SYMBOL.values():
                            word = num2words(word, lang='de', to='ordinal')
                            if idx > 0 and idx < (len(split_text) - 1):
                                if cleaned[idx - 1].endswith('m'):
                                    word += 'n'
                        # else:
                        #    word = word[:-1]

                # SPECIAL TERMS or CHARACTERS
                # TODO: because of the kind of split, only statements/terms with a single word
                #       are covered currently (e.g. "8/10" but not "8 / 10")
                elif tr == 'special':
                    for pat, tup_repl in self.SPECIAL_TRANSLITERATE.items():
                        if re.search(pat, word):
                            word = word.replace(tup_repl[0], tup_repl[1], 1)
                            ws = []
                            for w in word.split(' '):
                                if self.REX_DETECT_NUMBER.match(w):
                                    ws.append(self._transliterate_number(w))
                                else:
                                    ws.append(w)
                            if ws:
                                word = ' '.join(ws)

                # SOME MATH SYMBOLS
                elif tr == 'math_symbol':
                    for pat, repl in self.MATH_SYMBOL_TRANSLITERATE.items():
                        if pat in word:
                            # we need heuristics... TODO: having better idea?
                            if pat == 'x':
                                if len(word) > 1 and word.find(pat) == 0:
                                    continue
                            elif pat == '-':
                                if word == '--':
                                    continue
                                elif len(word) > 1 and \
                                        (len(word[:word.find(pat)]) > 1 or len(word[word.find(pat) + 1:]) > 1):
                                    continue

                            word = word.replace(pat, repl, 1)
                    ws = []
                    for w in word.split(' '):
                        if self.REX_DETECT_NUMBER.match(w):
                            ws.append(self._transliterate_number(w))
                        else:
                            ws.append(w)
                    if ws:
                        word = ' '.join(ws)

                # AUXILIARY SYMBOLS (e.g. brackets)
                elif tr == 'aux_symbol':
                    for pats, repl in self.AUX_SYMBOL_TRANSLITERATE.items():
                        if pats[0] in word:
                            word = word.replace(pats[0], repl.replace('_', self.SEP_ABBR_MASK))
                            if pats[1] in word:
                                word = word.replace(pats[1], self.SEP_ABBR_MASK)
                            else:
                                for fwd_idx in range(idx + 1, len(split_text)):
                                    if pats[1] in split_text[fwd_idx]:
                                        split_text[fwd_idx] = split_text[fwd_idx].replace(pats[1],
                                                                                          self.SEP_ABBR_MASK)
                                        break

            # REPLACE/MAP (remaining) SPECIFIC CHARACTERS
            for old, new in self.replace.items():
                word = word.replace(old, new)

            # REPLACE/MAP any abbreviations or short forms
            for short, long in self.ABBREVIATION.items():
                if long not in word:
                    # invariant: cover also abbreviation with '.' at the end
                    if (short + '.') == word or short == word:
                        word = long

            # EXPAND UNITS SEPARATED from numbers
            w_unit = word
            # TODO: remove non-alphanum for mapping to transliterations, e.g. boundary chars like "!" or "?"
            if w_unit.endswith('.'):
                w_unit = w_unit[:-1]
            w_unit = w_unit.replace(self.SEP_ABBR_MASK, '_')

            if w_unit in self.UNIT_0.keys():
                word = self.UNIT_0[w_unit]
                if idx > 0 and cleaned[idx - 1] == ('eins'):
                    cleaned[idx - 1] = 'ein'
            elif w_unit in self.UNIT_2.keys():
                word = self.UNIT_2[w_unit]
                # invariant: one <unit>
                if idx > 0 and (cleaned[idx - 1] == ('eins')
                                or cleaned[idx - 1] == ('ein')
                                or cleaned[idx - 1] == ('eine')
                                or cleaned[idx - 1] == ('einer')
                                or cleaned[idx - 1] == ('einen')
                                or cleaned[idx - 1] == ('einem')):
                    word = word[:-2]
                    cleaned[idx - 1] = 'eine'

            elif w_unit in self.UNIT_1.keys():
                word = self.UNIT_1[w_unit]
                # invariant: one <unit>
                if idx > 0 and (cleaned[idx - 1] == ('eins')
                                or cleaned[idx - 1] == ('ein')
                                or cleaned[idx - 1] == ('eine')
                                or cleaned[idx - 1] == ('einer')
                                or cleaned[idx - 1] == ('einen')
                                or cleaned[idx - 1] == ('einem')):
                    word = word[:-1]
                    cleaned[idx - 1] = 'eine'

            # EXPAND (remaining) NUMBERS
            num_match = self.REX_DETECT_NUMBER.match(word)
            if num_match:

                # EXPAND UNITS ATTACHED to numbers
                w_unit = word[num_match.end():]
                invariant_one = False
                if w_unit:
                    if w_unit.endswith('.'):
                        w_unit = w_unit[:-1]

                    if w_unit in self.UNIT_0.keys():
                        w_unit = self.UNIT_0[w_unit]
                    elif w_unit in self.UNIT_2.keys():
                        w_unit = self.UNIT_2[w_unit]
                        # invariant: one <unit>
                        if num_match.group(0).replace('+', '').replace('-', '') in ['1', '1.0', '1.00']:
                            w_unit = w_unit[:-2]
                            invariant_one = True
                    elif w_unit in self.UNIT_1.keys():
                        w_unit = self.UNIT_1[w_unit]
                        # invariant: one <unit>
                        if num_match.group(0).replace('+', '').replace('-', '') in ['1', '1.0', '1.00']:
                            w_unit = w_unit[:-1]
                            invariant_one = True
                    w_unit = ' ' + w_unit
                    word = num_match.group(0)

                if invariant_one:
                    word = 'eine'
                else:
                    word = self._transliterate_number(word)
                word += w_unit

            cleaned.append(word)
            idx += 1

        text = ' '.join(cleaned)
        return self.REX_DETECT_WHITESPACE_SEQ.sub(' ', text.replace(self.SEP_ABBR_MASK, self.sep_abbreviation))

    def _transliterate_number(self, number: str) -> str:
        from num2words import num2words

        # IMPORTANT NOTE: GERMAN version means 1000's marks is "."
        # and decimal point is "," (the opposite of in English)

        # invariant: 1000's mark(s) AND floating point number
        if number.count(',') == 1 and number.count('.') >= 1:
            number = number.replace('.', '')

        try:
            # floating number only
            if number.count(',') == 1:
                number = number.replace(',', '.')
                word = num2words(float(number), lang='de', to='cardinal').lower()
            # 1000's marks only
            elif number.count('.') >= 1:
                number = number.replace('.', '')
                word = num2words(int(number), lang='de', to='cardinal').lower()
            # integer only
            else:
                word = num2words(int(number), lang='de', to='cardinal').lower()
        except ValueError:
            # ignore HERE: mixed numbers are handled further down in the pipeline!
            word = number

        return word


text = "Hr. Thorsten Müller (in seiner Rolle als Chef-Sprecher) " \
       "hat alle Sätze von A-Z selber 2x oder sogar x-mal, ohne dafür" \
       " auch nur einen € zu bekommen, eingesprochen -- insgesamt fast 20h. " \
       "Dafür wird er bald Mio. Hearts & Minds gewinnen, 1000 Dank!"

text = '(1/10) mit (vielleicht 1000) Möglichkeiten'

text = 'Am 23. April 1819 waren die geschäfte zu. Von 23:54h an und vielleicht auch zum Fr. 14. Mai 2020. Zeitstempel: 01h:10m:20sec gehts los, wiegt fast 20 kg!'

text =  "Das Salz Calciumchlorid wird von Ca2+ und Cl− gebildet."

#print('ORIGINAL:', text)
#print('TRANSLITERATION:', german_phoneme_cleaners(text))
