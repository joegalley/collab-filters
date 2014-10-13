import sys
from abc import ABCMeta, abstractmethod
from collections import defaultdict


# abstract base class - must be extended by all
# types of collaborative filter classes
class CollaborativeFilter(metaclass=ABCMeta):
    @abstractmethod
    def readTrainingData(self, data_file):
        with open(data_file, "r") as f:
            tr_data = [line.strip().split("\t") for line in f]
        return tr_data

    @abstractmethod
    def readTestData(self, data_file):
        with open(data_file, "r") as f:
            tr_data = [line.strip().split("\t") for line in f]
        return tr_data

    @abstractmethod
    def calculate(self):
        pass



class Average(CollaborativeFilter):
    """Collaborative filter based on users' average ratings. By definition,
     is independent of user's identity.
    """

    training_data = None
    test_data = None    

    
    def readTrainingData(self, data):
        self.training_data = super(Average, self).readTrainingData(data)
        flat_list = ([int(x) for x in [i for row in self.training_data for i in row]])
        users = flat_list[0::4]
        ratings = flat_list[2::4]
        z = zip(users, ratings)
        users_to_ratings = defaultdict(list)
        for k, v in z:
            users_to_ratings[k].append(v)
        '''
        for k, v in users_to_ratings.items():
            for i in v:
                print(k, " ", i)      
        '''
        self.training_data = users_to_ratings
                

    def readTestData(self, data):
        self.test_data = super(Average, self).readTrainingData(data)
        flat_list = ([int(x) for x in [i for row in self.test_data for i in row]])
        users = flat_list[0::4]
        ratings = flat_list[2::4]
        z = zip(users, ratings)
        users_to_ratings = defaultdict(list)
        for k, v in z:
            users_to_ratings[k].append(v)

        self.test_data = users_to_ratings
        
    def calculateError(self):
        return RSME(self.training_data, self.test_data)

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

def RSME(training_data, test_data):
    for k_training, v_training in training_data.items():
        for k_test, v_test in test_data.items():
            if(k_training == k_test):
                for rating_training, rating_test in zip(v_training, v_test):
                    print((rating_training - rating_test) ** 2)


                
        
             
    pass