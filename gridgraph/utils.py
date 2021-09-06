import os
import csv
import logging


def param_loader(path):
    '''Read a csv containing value/spread pairs of hyperparameters into a
    dictionary.'''
    params = {}
    with open(path, newline='', encoding='utf-8-sig') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            params[row['param']] = float(row['value'])
    return params


def start_logging(loc, identifier=None):
    if not os.path.exists(loc):
        os.makedirs(loc)
    loc = os.path.join(loc, 'log.txt')
    set_logger(loc)
    if identifier is None:
        logging.info('Began logging to: %s', loc)
    else:
        logging.info('Began logging %s to: %s', identifier, loc)


def set_logger(log_path):
    """From https://github.com/cs230-stanford/cs230-code-examples
    Set the logger to log info in terminal and file `log_path`.
    In general, it is useful to have a logger so that every output to the
    terminal is saved in a permanent file. Here we save it to
    `model_dir/train.log`.

    Example:
    ```
    logging.info("Starting training...")
    ```
    Args:
        log_path: (string) where to log
    """
    logger = logging.getLogger()
    logger.handlers = []
    logger.setLevel(logging.INFO)

    if not logger.handlers:
        # Logging to a file
        file_handler = logging.FileHandler(log_path)
        file_handler.setFormatter(
            logging.Formatter('%(asctime)s:%(levelname)s: %(message)s'))
        logger.addHandler(file_handler)

        # Logging to console
        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(logging.Formatter('%(message)s'))
        logger.addHandler(stream_handler)
