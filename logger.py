import logging
from pathlib import Path


class Logger:
    """
    Logger aligned with AA-CLIP: standard logging format `LEVEL:logger:message`
    with outputs to both log file and console.
    """

    def __init__(self, log_path: str):
        self.log_path = Path(log_path)
        self.log_path.parent.mkdir(parents=True, exist_ok=True)

        formatter = logging.Formatter("%(levelname)s:%(name)s:%(message)s")

        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.INFO)

        # Clear existing handlers to avoid duplicates.
        root_logger = logging.getLogger()
        for handler in root_logger.handlers[:]:
            root_logger.removeHandler(handler)
        for handler in self.logger.handlers[:]:
            self.logger.removeHandler(handler)

        file_handler = logging.FileHandler(self.log_path, mode="a", encoding="utf-8")
        file_handler.setFormatter(formatter)

        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)

        self.logger.addHandler(file_handler)
        self.logger.addHandler(console_handler)

    def info(self, msg: str):
        self.logger.info(msg)

    def warning(self, msg: str):
        self.logger.warning(msg)

    def error(self, msg: str):
        self.logger.error(msg)

    def debug(self, msg: str):
        self.logger.debug(msg)
