from abc import ABC, abstractmethod


class BaseImageRepresentation(ABC):
    name: str

    @abstractmethod
    def transform(self, x, y, z):
        pass

    @staticmethod
    def get_by_name(name):
        classes = {cls.name: cls for cls in BaseImageRepresentation.__subclasses__()}
        return classes.get(name)
