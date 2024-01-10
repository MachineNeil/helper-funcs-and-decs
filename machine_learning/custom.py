import re

NUMBERS_REGEX = re.compile(r'\d+')

EMAIL_REGEX = re.compile(
    r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,7}\b')

PHONE_REGEX = re.compile(r'\d+')  # PLACEHOLDER

NON_ASCII_REGEX = re.compile(r'[^\x00-\x7F]+')

JUNK_REGEX = re.compile(
    r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
)

SCAPE_REGEX = re.compile(r'\s+')

WHITESPACE_REGEX = re.compile(r'\s+')

LIST_REGEX = re.compile(r'\w[.)]\s*')

EMOJI_REGEX = re.compile(
    "["
    "\U0001F1E0-\U0001F1FF"
    "\U0001F300-\U0001F5FF"
    "\U0001F600-\U0001F64F"
    "\U0001F680-\U0001F6FF"
    "\U0001F700-\U0001F77F"
    "\U0001F780-\U0001F7FF"
    "\U0001F800-\U0001F8FF"
    "\U0001F900-\U0001F9FF"
    "\U0001FA00-\U0001FA6F"
    "\U0001FA70-\U0001FAFF"
    "\U00002702-\U000027B0"
    "\U000024C2-\U0001F251"
    "]+"
)

IMAGE_REGEX = re.compile(r'\b\w+\.(png|jpg|gif)\b')

URL_REGEX = re.compile(r'\d+')  # Placeholder
