import logging
import constants
from pythonjsonlogger import jsonlogger

logger = logging.getLogger(__name__)

text_fh = logging.FileHandler(
    constants.FILENAME_TEXT, encoding='utf-8')
json_fh = logging.FileHandler(
    constants.FILENAME_JSON, encoding='utf-8')

text_fmt = logging.Formatter(constants.FORMAT_TEXT)
json_fmt = jsonlogger.JsonFormatter(constants.FORMAT_JSON,
                                    rename_fields={'name': 'loggername',
                                                   'levelname': 'severity',
                                                   'lineno': 'line_number',
                                                   'asctime': 'timestamp'},
                                    json_indent=4
                                    )

text_fh.setFormatter(text_fmt)
json_fh.setFormatter(json_fmt)

logger.addHandler(text_fh)
logger.addHandler(json_fh)

logger.setLevel(logging.DEBUG)
