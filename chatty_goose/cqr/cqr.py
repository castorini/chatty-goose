from typing import Optional
from abc import abstractmethod
import logging


__all__ = ["ConversationalQueryRewriter"]


class ConversationalQueryRewriter:
    """Base conversational query reformulation class"""

    def __init__(self, name: str, verbose: bool = False):
        self.name = name
        self.turn_id: int = -1
        self.total_latency: float = 0.0
        self.verbose: bool = verbose

    @abstractmethod
    def rewrite(self, query: str, context: Optional[str] = None) -> str:
        """Rewrite original query text"""
        raise NotImplementedError

    def reset_history(self):
        """Reset conversation history for model"""
        if self.verbose and self.turn_id > -1:
            turns = self.turn_id + 1
            logging.info(
                "Resetting {} after {} turns (average reformulation latency {:.4f}s)".format(
                    self.name, turns, self.total_latency / (turns)
                )
            )
        self.turn_id = -1
        self.total_latency = 0
