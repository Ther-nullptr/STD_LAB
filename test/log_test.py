import logging

if __name__ == '__main__':
    logger = logging.getLogger(__name__)
    logger.setLevel(level = logging.INFO)
    formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(filename)s : %(lineno)s line - %(message)s")

    file_handler = logging.FileHandler(filename="debug/debug.log")
    file_handler.setFormatter(formatter)
    file_handler.setLevel(level = logging.INFO)
    logger.addHandler(file_handler)

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    stream_handler.setLevel(level = logging.INFO)
    logger.addHandler(stream_handler)

    logger.info('hello world xxx yyy')