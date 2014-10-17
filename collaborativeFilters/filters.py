import sys
from abc import ABCMeta, abstractmethod
from collections import defaultdict
from math import sqrt
from collections import Counter

DEBUG = True

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
        items = flat_list[1::4]
        ratings = flat_list[2::4]
        z = zip(items, ratings)
        items_to_ratings = defaultdict(list)
        for k, v in z:
            items_to_ratings[k].append(v)

        for k in items_to_ratings:
            print(k, items_to_ratings[k])
       
        self.training_data = items_to_ratings
                

    def readTestData(self, data):
        self.test_data = super(Average, self).readTrainingData(data)
        flat_list = ([int(x) for x in [i for row in self.test_data for i in row]])
        items = flat_list[1::4]
        ratings = flat_list[2::4]
        z = zip(items, ratings)
        items_to_ratings = defaultdict(list)
        for k, v in z:
            items_to_ratings[k].append(v)

        for k in items_to_ratings:
            print(k, items_to_ratings[k])

        self.test_data = items_to_ratings

    def calculateError(self):
        return RSME(self.training_data, self.test_data)

    def calculate():
        pass
        
        


class UserEucledian(CollaborativeFilter):

    training_data = None

    def readTrainingData(self, data):
        self.training_data = super(UserEucledian, self).readTrainingData(data)

        user_col = 0
        item_col = 1
        rating_col = 2
        timestamp_col = 3

        # loop through all combinations of 2 users, not matching a user with himself
        first_user_idx = 0
        sum_diff_sq_user = 0
        for first_user in self.training_data:
            # sum of squared differences in rating for first_user, second_user
            
            second_user_idx = 0
            sum_diff_sq_item = 0
            for second_user in self.training_data:
                # true if the user is not himself, and he rated the same item as the second user                
                if self.training_data[first_user_idx][user_col] != self.training_data[second_user_idx][user_col] and self.training_data[first_user_idx][item_col] == self.training_data[second_user_idx][item_col]:
                    # the squared difference between first_user and second_user's rating for the same item
                    diff_sq = (int(self.training_data[first_user_idx][rating_col]) - int(self.training_data[second_user_idx][rating_col])) ** 2
                    sum_diff_sq_item += diff_sq                    
                    if(DEBUG):
                        print("User: ", self.training_data[first_user_idx][user_col], " Item: ", self.training_data[first_user_idx][item_col], " Rating:", self.training_data[first_user_idx][rating_col], "\n", "User:", self.training_data[second_user_idx][user_col], " Item: ", self.training_data[second_user_idx][item_col], " Rating:", self.training_data[second_user_idx][rating_col], "Difference^2: ", str(diff_sq))
                second_user_idx += 1  
            print(sum_diff_sq_item)
            sum_diff_sq_user += sum_diff_sq_item

            first_user_idx += 1
        


    def readTestData(self, tr_data):
        pass
    def run():
        pass
    def calculate():
        pass



class UserPearson(CollaborativeFilter):
    def readTrainingData(self, tr_data):
        print("in child")
    def readTestData(self, tr_data):
        pass
    def calculate():
        pass


class ItemCosine(CollaborativeFilter):
    training_data = None
    cosine_similarities = {}

    class ItemRatingUser():
        item = None
        rating = None
        user = None

        def __init__(self, item, rating, user):
            self.item = item
            self.rating = rating
            self.user = user

        pass





    def readTrainingData(self, data):
        self.training_data = super(ItemCosine, self).readTrainingData(data)
        
        user_col = 0
        item_col = 1
        rating_col = 2
        timestamp_col = 3

        items = [row[item_col] for row in self.training_data] 
        ratings = [row[rating_col] for row in self.training_data] 
        users = [row[user_col] for row in self.training_data] 

        item_to_ratings = defaultdict(list)

        z = zip(items, ratings, users)

        print("ITEM -- RATING -- USER")
        for item, rating, user in z:
            print(item, "   ", rating, "       ", user)
            item_to_ratings[item].append([rating, user])

        item_pairs_to_ratings = {}
        for item1 in item_to_ratings.items():
            for item2 in item_to_ratings.items():
                item_pairs_to_ratings[item1[0], item2[0]] = [None]
                print("here", item1[1])

                
                    

                # don't compare the same item
                if(item1[0] != item2[0]):
                    print("past", item1, item2)



                    # remove 0 ratings
                    for rating in item1[1]:
                        if(int(rating[0]) == 0):
                            item1[1].remove(rating)
                    for rating in item2[1]:
                        if(int(rating[0]) == 0):
                            item2[1].remove(rating)
         
                    # LEFT OFF HERE #

                    print("after removed:", item1[1], item2[1])

                    for rating in item1[1]:
                        if rating[1] not in item2[1][1]:
                            print(rating, "not in 1")

                    # only add items which have been rated by the same set of users
                    users_who_rated_item1 = []
                    for rating_and_user in item1[1]:
                        users_who_rated_item1.append(rating_and_user[1])
                    users_who_rated_item2 = []
                    for rating_and_user in item2[1]:
                        users_who_rated_item2.append(rating_and_user[1])

                    users_who_rated_both_items = set(users_who_rated_item1).intersection(users_who_rated_item2)

                    num_rated_item1_intersect_num_rated_both = set(users_who_rated_item1).intersection(users_who_rated_both_items)
                    num_rated_item2_intersect_num_rated_both = set(users_who_rated_item2).intersection(users_who_rated_both_items)

                    print("num1 in", num_rated_item1_intersect_num_rated_both)
                    print("num2 in", num_rated_item2_intersect_num_rated_both)

                    if len(num_rated_item1_intersect_num_rated_both) > 0 or len(num_rated_item2_intersect_num_rated_both) > 0:
                        print("ADDING", item1[0],":", item1[1], item2[0], ":", item2[1])
                        item_pairs_to_ratings[item1[0], item2[0]] = [item1[1], item2[1]]

                    '''
                    valid_rating = False

                    for i in range(0, len(item1[1])):
                        if(item1[1][i][1] not in users_who_rated_both_items):
                            print(item1[1][i][1], " IS NOT IN ", users_who_rated_both_items)
                            valid_rating = False
                        else:
                            print(item1[1][i][1], " IS IN ", users_who_rated_both_items)
                            valid_rating = True

                    for i in range(0, len(item2[1])):
                        if(item2[1][i][1] not in users_who_rated_both_items):
                            print(item2[1][i][1], " IS NOT IN ", users_who_rated_both_items)
                            valid_rating = False
                        else:
                            print(item2[1][i][1], " IS IN ", users_who_rated_both_items)
                            valid_rating = True                   

                    print("SDLKFJSLDKFJ", item1[1])
                    # enforce the above rule
                    if(valid_rating):
                        print("ADDING", item1[0], item2[0])
                        item_pairs_to_ratings[item1[0], item2[0]] = [item1[1], item2[1]]
                    else:
                        print("NOT ADDING", item1[0], item2[0])
                    '''
                    
        
        for item, item_and_user in item_pairs_to_ratings.items():
            if item[0] != None and item_and_user[0] != None and item[1] != None and item_and_user[1] != None and len(item_and_user[0]) == len(item_and_user[1]):
                print("here:", item[0], item_and_user[0], item[1], item_and_user[1])

                dot_prod = self.dotProduct(item_and_user[0], item_and_user[1])
                mag_vec0 = self.vecMagnitude(item_and_user[0])
                mag_vec1 = self.vecMagnitude(item_and_user[1])
                cosine_similarity = dot_prod/(mag_vec0 * mag_vec1)
                key_table = [item[0][0]]
                key_table.append(item[1][0])

                # convert to tuple so it can be hashed & used as dict key
                key_tuple = tuple(key_table)
                               
                self.cosine_similarities[key_tuple] = cosine_similarity

                if(DEBUG):
                    print("\nItems: ", item[0], item[1])
                    print("Dot product of ", item_and_user[0], " and ", item_and_user[1], " = ", dot_prod)
                    print("vec1 magnitude: ", mag_vec0)
                    print("vec2 magnitude: ", mag_vec1)
                    print("Cosine Similarity: ", cosine_similarity)

    def dotProduct(self, vec1, vec2):
        if len(vec1) != len(vec2):
            print("ERROR - vector length mismatch")
        
        else:
            # extract ratings from vec1, vec2 with index 0 - [rating, user][rating, user]...[rating, user]
            v1 = []
            for rating_user in vec1:
                v1.append(rating_user[0])

            v2 = []
            for rating_user in vec1:
                v2.append(rating_user[0])

            z = zip(v1, v2)
            dot_prod = 0
            for k in z:
                dot_prod += int(k[0]) * int(k[1]) 
            return dot_prod    

    def vecMagnitude(self, vec):

        # extract ratings from vec with index 0 - [rating, user][rating, user]...[rating, user]
        v = []
        for rating_user in vec:
            v.append(rating_user[0])


        print(v)
        mag = 0
        for i in v:
            mag += int(i) ** 2
        return sqrt(mag)
    
    def showCosineSimilarities(self):
        print("\nCOSINE SIMILARITIES:")
        for k in self.cosine_similarities:
            print(k, self.cosine_similarities[k])

    def readTestData(self, tr_data):
        pass
    def calculate():
        pass

class ItemAdjustedCosine(CollaborativeFilter):
    training_data = None
    usr_avg_data = None
    cosine_similarities = {}
    user_to_avg_rating = {}

    def readTrainingData(self, data):
        self.training_data = super(ItemAdjustedCosine, self).readTrainingData(data)

        self.user_to_avg_rating = self.getUsersAvgRatings(data)
        
        user_col = 0
        item_col = 1
        rating_col = 2
        timestamp_col = 3

        items = [row[item_col] for row in self.training_data] 
        ratings = [row[rating_col] for row in self.training_data] 
        users = [row[user_col] for row in self.training_data] 

        item_to_ratings_with_user = defaultdict(list)

        z = zip(items, ratings, users)

        for item, rating, user in z:
            item_to_ratings_with_user[item].append([rating, user])

        item_pairs_to_ratings = {}
        for item1 in item_to_ratings_with_user.items():
            for item2 in item_to_ratings_with_user.items():
                item_pairs_to_ratings[item1[0], item2[0]] = [None]
                # don't compare the same item
                if(item1[0] != item2[0]):
                    print("one", item1, "two", item2)
                    item_pairs_to_ratings[item1[0], item2[0]] = [item1[1], item2[1]]
                    
        
        for k, v in item_pairs_to_ratings.items():
            if(k[0] != None and v[0] != None and k[1] != None and v[1] != None and len(v[0]) == len(v[1])):
                print("here:", k[0], v[0], k[1], v[1])
                dot_prod = self.dotProduct(v[0], v[1])
                mag_vec0 = self.vecMagnitude(v[0])
                mag_vec1 = self.vecMagnitude(v[1])
                cosine_similarity = dot_prod/(mag_vec0 * mag_vec1)
                key_table = [k[0][0]]
                key_table.append(k[1][0])

                # convert to tuple so it can be hashed & used as dict key
                key_tuple = tuple(key_table)
                               
                self.cosine_similarities[key_tuple] = cosine_similarity

                if(DEBUG):
                    print("\nItems: ", k[0], k[1])
                    print("Dot product of ", v[0], " and ", v[1], " = ", dot_prod)
                    print("vec1 magnitude: ", mag_vec0)
                    print("vec2 magnitude: ", mag_vec1)
                    print("Cosine Similarity: ", cosine_similarity)

    def dotProduct(self, vec1, vec2):
        print("v1: ", vec1, "vec2 ", vec2)

        user1 = []
        for user in vec1:
            user1.append(user[1])

        user2 = []
        for user in vec2:
            user2.append(user[1])

        if len(set(user1)) != 1 or len(set(user2)) != 1:
            print("ERROR - not comparing rating from the same user")

        user1_ratings_vec = []

        user2_ratings_vec = []

        for rating in vec1:
            user1_ratings_vec.append(rating[0])

        for rating in vec2:
            user2_ratings_vec.append(rating[0])



        if len(user1_ratings_vec) != len(user2_ratings_vec):
            print("ERROR - vector length mismatch")
        else:
            z = zip(user1_ratings_vec, user2_ratings_vec)
            dot_prod = 0
            for k in z:
                print("a", k[0], "b", k[1], "c", self.user_to_avg_rating[user1[0]], "d", self.user_to_avg_rating[user2[0]])
                dot_prod += float(int(k[0]) - self.user_to_avg_rating[user1[0]]) * float(float(k[1]) - self.user_to_avg_rating[user2[0]]) 
            return dot_prod    

    def vecMagnitude(self, vec):
        mag = 0

        for i in vec:
            print("SDF", i, self.user_to_avg_rating[i[1]])
            mag += float(int(i[0]) - self.user_to_avg_rating[i[1]]) ** 2
        return sqrt(mag)

    def getUsersAvgRatings(self, data):
        self.usr_avg_data = super(ItemAdjustedCosine, self).readTrainingData(data)

        user_col = 0
        item_col = 1
        rating_col = 2
        timestamp_col = 3

        users = [row[user_col] for row in self.usr_avg_data] 
        ratings = [row[rating_col] for row in self.usr_avg_data] 

        users_to_ratings = defaultdict(list)

        z = zip(users, ratings)

        for k, v in z:
            users_to_ratings[k].append(v)    

        user_to_avg_rating = {}
        for user in users_to_ratings:
            user_to_avg_rating[user] = None
            avg_rating = 0
            num_ratings = 0
            print("USER :", user, "RATINGS: ", users_to_ratings[user])
            for rating in users_to_ratings[user]:
                avg_rating += int(rating)
                num_ratings += 1
            user_to_avg_rating[user] = avg_rating / num_ratings

        for user in user_to_avg_rating:
            print("USER: ", user, "AVG RATING: ", user_to_avg_rating[user])

        return user_to_avg_rating


    
    def showCosineSimilarities(self):
        print("\nCOSINE SIMILARITIES:")
        for k in self.cosine_similarities:
            print(k, self.cosine_similarities[k])

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
    sum_squared_dif = 0
    num_obvs = 0

    for k_training, v_training in training_data.items():
        print("k_training :", k_training, "v_training: ", v_training)
        for k_test, v_test in test_data.items():
            print("k_test: ", k_test, "v_test: ", v_test)
            if(k_training == k_test):
                print("Yes")
                for rating_training, rating_test in zip(v_training, v_test):
                    print("tr:", rating_training, "test", rating_test)
                    sum_squared_dif += (rating_training - rating_test) ** 2
                    num_obvs += 1

    rmse = sqrt(sum_squared_dif/num_obvs)
    return rmse
                
        
             
    pass