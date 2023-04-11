import logging
from rich.logging import RichHandler
import numpy as np

np.set_printoptions(precision=4, suppress=True, sign=" ")


class Logger:

    FORMAT = "%(message)s"

    def __init__(self, name: str, log_level="DEBUG"):
        self.log = logging.getLogger(name)
        self.level = log_level

        # File handler
        # f_handler = logging.FileHandler("./assignment4/logs/icp_log.log", mode="w")
        # f_handler.setLevel("INFO")
        # f_handler.setFormatter(logging.Formatter(fmt=Logger.FORMAT, datefmt="[%X]"))
        # log.addHandler(f_handler)

        # Simple Stream handler
        # s_handler = logging.StreamHandler()
        # s_handler.setLevel("INFO")
        # s_handler.setFormatter(logging.Formatter(fmt=Logger.FORMAT, datefmt="[%X]"))
        # self.log.addHandler(s_handler)

        # Rich stream handler
        r_handler = RichHandler(rich_tracebacks=True)
        r_handler.setLevel(self.level)
        r_handler.setFormatter(logging.Formatter(fmt=Logger.FORMAT, datefmt="[%X]"))
        self.log.addHandler(r_handler)
        self.log.setLevel(self.level)
