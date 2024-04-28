from abc import ABC, abstractmethod




'''
    an interface for TKG data loader
'''
class DataLoaderIface(ABC):
    
    @abstractmethod
    def read_entities():
        pass

    @abstractmethod
    def read_relations():
        pass

    @abstractmethod
    def read_triplets():
        pass

    @abstractmethod
    def build_kg():
        pass

    @abstractmethod
    def get_h2t():
        pass

    @abstractmethod
    def get_paths():
        pass

    @abstractmethod
    def load_data():
        pass