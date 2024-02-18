import os
import csv
from datetime import datetime
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from src.constants import cfg

def log_to_csv(question, answer):

    log_dir, log_file = cfg.STORAGE.HISTORY_DIR, cfg.STORAGE.HISTORY_FILE
    # Ensure log directory exists, create if not
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    # Construct the full file path
    log_path = os.path.join(log_dir, log_file)

    # Check if file exists, if not create and write headers
    if not os.path.isfile(log_path):
        with open(log_path, mode='w', newline='', encoding='utf-8') as file:
            writer = csv.writer(file)
            writer.writerow(["timestamp", "question", "answer"])

    # Append the log entry
    with open(log_path, mode='a', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        writer.writerow([timestamp, question, answer])

# def backtranslate_augment(text, target_language="vi", num_augmentations=1):
#    """Augments text using backtranslation with the specified target language.

#    Args:
#        text: The text to augment.
#        target_language: The language to translate the text to and back from.
#        num_augmentations: The number of augmented versions to generate.

#    Returns:
#        A list of augmented text versions.
#    """

#    translator = Translator()
#    augmented_texts = []

#    for _ in range(num_augmentations):
#        # Translate to target language
#        translated_text = translator.translate(text, dest=target_language).text

#        # Translate back to original language
#        backtranslated_text = translator.translate(translated_text, src=target_language).text

#        augmented_texts.append(backtranslated_text)

#    return augmented_texts