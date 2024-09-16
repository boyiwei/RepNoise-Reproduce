import os
import time

from googleapiclient import discovery
from googleapiclient.errors import HttpError

from loguru import logger

API_KEY = os.environ.get('PERSPECTIVE_API_KEY')

client = discovery.build(
    "commentanalyzer",
    "v1alpha1",
    developerKey=API_KEY,
    discoveryServiceUrl="https://commentanalyzer.googleapis.com/$discovery/rest?version=v1alpha1",
    static_discovery=False,
)


def get_toxicity_score(
    text: str
):
    analyze_request = {
        'comment': {'text': text},
        'requestedAttributes': {'TOXICITY': {}},
        'spanAnnotations': False,
        'languages': ['en']
    }
    response = None
    max_retries = 5
    current_retries = 0
    while not response:
        try:
            response = client.comments().analyze(body=analyze_request).execute()
        except HttpError as e:
            if "COMMENT_EMPTY" in str(e):
                logger.warning("Comment is empty, skipping...")
                return None
            print(f'Perspective API threw an error: {e}\n Retrying...')
            time.sleep(10)
            current_retries += 1
            if current_retries > max_retries:
                logger.error(f'Perspective API failed after {max_retries} retries. Exiting...')
                return None

    return response['attributeScores']['TOXICITY']['summaryScore']['value']
