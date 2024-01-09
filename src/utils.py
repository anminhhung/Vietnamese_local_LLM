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