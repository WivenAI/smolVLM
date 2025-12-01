from abc import ABC, abstractmethod

class BaseEvaluator(ABC):
    @abstractmethod
    def accuracy(self, results):
        """Common interface - calls the specific calculate_accuracy"""
        pass