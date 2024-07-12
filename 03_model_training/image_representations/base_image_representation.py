from abc import ABC, abstractmethod


class BaseImageRepresentation(ABC):
    name: str

    @abstractmethod
    def transform(self, x, y, z):
        pass
