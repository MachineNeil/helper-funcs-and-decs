import contractions
import custom
from typing import Optional, Any
import string
import nltk
from bs4 import BeautifulSoup
import unicodedata


def set_case(text: str, case: str) -> str:
    if case == 'lowercase':
        return text.lower()
    elif case == 'uppercase':
        return text.upper()


def remove_numbers(text: str, number_replacement: str) -> str:
    return custom.NUMBERS_REGEX.sub(number_replacement, text)


def remove_punctuation(text: str, punctuation: Optional[str] = None, special_punctuation: Optional[str] = None) -> str:
    if punctuation is None:
        punctuation = string.punctuation
    if special_punctuation is not None:
        punctuation += special_punctuation
    return text.translate(str.maketrans('', '', punctuation))


def remove_special_characters(text: str, special_characters: str) -> str:
    return text.translate(str.maketrans('', '', special_characters))


def remove_non_alpha(text: str, alphanum: bool) -> str:
    if alphanum:
        return ''.join(filter(lambda char: char.isalnum() or char == " ", text))
    else:
        return ''.join(filter(lambda char: char.isalpha() or char == " ", text))


def remove_stopwords(text: str, language: str, stopwords: list = None, special_stopwords: list = None) -> str:
    if stopwords is None:
        stopwords = nltk.corpus.stopwords.words(language)
    if special_stopwords is not None:
        stopwords += special_stopwords
    return ' '.join([word for word in text.split() if word not in set(stopwords)])


def remove_email(text: str, replacement: str) -> str:
    return custom.EMAIL_REGEX.sub(replacement, text)


def remove_phone(text: str, replacement: str) -> str:
    return custom.PHONE_REGEX.sub(replacement, text)


def remove_html(text: str) -> str:  # Falta replace con <HTML>
    soup = BeautifulSoup(text, 'lxml')
    return soup.get_text(strip=True)


def word_tokens(text: str, language: str) -> list:
    return nltk.word_tokenize(text, language)


def sentence_tokens(text: str, language: str) -> list:
    return nltk.sent_tokenize(text, language)


def normalize_nonascii(text: str) -> str:
    return unicodedata.normalize('NFKD', text).encode(
        'ascii', 'ignore').decode('utf-8', 'ignore')


def remove_nonascii(text: str, replacement) -> str:
    return custom.NON_ASCII_REGEX.sub(replacement, text)


def remove_url(text: str, replacement: str) -> str:
    return custom.URL_REGEX.sub(replacement, text)


def expand_contractions(text: str) -> str:
    return ' '.join([contractions.fix(word) for word in text.split()])


def remove_escape_characters(text: str, replacement: str) -> str:
    return custom.SCAPE_REGEX.sub(replacement, text)


def lemmatize(text: str, lemmatizer: Any):
    if lemmatizer is None:
        lemmatizer = nltk.stem.WordNetLemmatizer()
    return ' '.join([lemmatizer.lemmatize(word) for word in text.split()])


def stemming(text: str, stemmer: Any, language: str):
    if stemmer is None:
        stemmer = nltk.stem.SnowballStemmer(
            language=language)
    return ' '.join([stemmer.stem(word) for word in text.split()])


def remove_short_words(text: str, minimum_word_length: int) -> str:
    return ' '.join([word for word in text.split() if len(word) >= minimum_word_length])


def remove_images(text: str, replacement: str) -> str:
    return custom.IMAGE_REGEX.sub(replacement, text)


def remove_duplicate_whitespaces(text: str) -> str:
    return custom.WHITESPACE_REGEX.sub(' ', text.strip())


def remove_lists(text: str, replacement: str) -> str:
    return custom.LIST_REGEX.sub(replacement, text)


def remove_emojis(text: str, replacement: str) -> str:
    return custom.EMOJI_REGEX.sub(replacement, text)


def remove_names(text: str, replacement: str) -> str:
    tagged = nltk.tag.pos_tag(text.split())
    return ' '.join([replacement if position == 'NNP' else word for word, position in tagged])
