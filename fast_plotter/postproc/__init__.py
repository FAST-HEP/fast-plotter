import logging
from .functions import open_many
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


__all__ = ["open_many"]
