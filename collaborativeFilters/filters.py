import sys
from abc import ABCMeta, abstractmethod


# abstract base class - must be extended by all
# types of collaborative filter classes
class CollaborativeFilter(metaclass=ABCMeta):
    @abstractmethod
    def readTrainingData(self, tr_data_file):
        with open(tr_data_file, "r") as f:
            tr_data = [line.strip().split("\t") for line in f]
        return tr_data
    @abstractmethod
    def readTestData(self, test_data):
        pass

    @abstractmethod
    def calculate(self):
        pass



class Average(CollaborativeFilter):
    """Collaborative filter based on users' average ratings. By definition,
     is independent of user's identity.

    Attributes:
      attr1 (str): Description of `attr1`.
      attr2 (list of str): Description of `attr2`.
      attr3 (int): Description of `attr3`.

    """

    training_data = None
    training_data_avg = None
    test_data = None    
    test_data_avg = None
    
    def readTrainingData(self, data):
        self.training_data = super(Average, self).readTrainingData(data)
        flat_list = ([int(x) for x in [i for row in self.training_data for i in row]])
        ratings = flat_list[2::4]
        self.training_data_avg = sum(ratings) / len(ratings)
        print(self.training_data_avg)
        

    def readTestData(self, data):
        self.test_data = super(Average, self).readTestData(data)
        flat_list = ([int(x) for x in [i for row in self.test_data for i in row]])
        ratings = flat_list[2::4]
        self.test_data_avg = sum(ratings) / len(ratings)
        print(self.test_data_avg)


    def calculate():
        pass
        


class UserEucledian(CollaborativeFilter):
    def readTrainingData(self, tr_data):
        print("in child")
    def readTestData(self, tr_data):
        pass
    def run():
        pass


class UserPearson(CollaborativeFilter):
    def readTrainingData(self, tr_data):
        print("in child")
    def readTestData(self, tr_data):
        pass
    def calculate():
        pass


class ItemCosine(CollaborativeFilter):
    def readTrainingData(self, tr_data):
        print("in child")
    def readTestData(self, tr_data):
        pass
    def calculate():
        pass

class ItemAdjustedCosine(CollaborativeFilter):
    def readTrainingData(self, tr_data):
        print("in child")
    def readTestData(self, tr_data):
        pass
    def calculate():
        pass


class SlopeOne(CollaborativeFilter):
    def readTrainingData(self, tr_data):
        print("in child")
    def readTestData(self, tr_data):
        pass
    def calculate():
        pass