import nltk

nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('punkt')

import logging

# Set up logger
logger = logging.getLogger(__name__)

# Set logger level to ERROR
logger.setLevel(logging.ERROR)

# Create console handler and set its level to ERROR
ch = logging.StreamHandler()
ch.setLevel(logging.ERROR)

# Create formatter and add it to the handler
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
ch.setFormatter(formatter)

# Add handler to logger
logger.addHandler(ch)
