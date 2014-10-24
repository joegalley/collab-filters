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

class UserItemRating(dict):
    def __getitem__(self, item):
        try:
            return dict.__getitem__(self, item)
        except KeyError:
            value = self[item] = type(self)()
            return value




class Average(CollaborativeFilter):
    training_data = None
    test_data = None

    user_item_rating_training = None
    user_item_rating_test = None

    training_results = None
    test_results = None

    def readTrainingData(self, data):
        self.training_data = super(Average, self).readTrainingData(data)

        user_col = 0
        item_col = 1
        rating_col = 2
        timestamp_col = 3

        items = [row[item_col] for row in self.training_data] 
        ratings = [row[rating_col] for row in self.training_data] 
        users = [row[user_col] for row in self.training_data] 

        z = zip(users, items, ratings)

        user_item_rating = UserItemRating()
        for user, item, rating in z:
            user_item_rating[user][item] =  rating

        item2rating = defaultdict(list)
        for k, v in z:
            items_to_ratings[k].append(v)


        for user in user_item_rating:
            for item in user_item_rating[user]:
                for rating in user_item_rating[user][item]:
                    item2rating[item].append(int(rating))

        self.user_item_rating_training = user_item_rating

        item2avg_rating = {}
        for item in item2rating:
            item_rating = 0 
            times_rated = 0
            for rating in item2rating[item]:
                if rating != 0:
                    item_rating += rating
                    times_rated += 1
            item2avg_rating[item] = item_rating / times_rated
        
        self.training_results = item2avg_rating

    def readTestData(self, data):
        self.training_data = super(Average, self).readTrainingData(data)

        user_col = 0
        item_col = 1
        rating_col = 2
        timestamp_col = 3

        items = [row[item_col] for row in self.training_data] 
        ratings = [row[rating_col] for row in self.training_data] 
        users = [row[user_col] for row in self.training_data] 

        z = zip(users, items, ratings)

        user_item_rating = UserItemRating()
        for user, item, rating in z:
            user_item_rating[user][item] =  rating

        item2rating = defaultdict(list)
        for k, v in z:
            items_to_ratings[k].append(v)

        for user in user_item_rating:
            for item in user_item_rating[user]:
                for rating in user_item_rating[user][item]:
                    item2rating[item].append(int(rating))

        self.user_item_rating_test = user_item_rating

        item2avg_rating = {}
        for item in item2rating:
            item_rating = 0 
            times_rated = 0
            for rating in item2rating[item]:
                if rating != 0:
                    item_rating += rating
                    times_rated += 1
            item2avg_rating[item] = item_rating / times_rated
        
        self.test_results = item2avg_rating

    def RMSE(self):

        squred_differences = 0
        rating_diffs = []
        for user in self.user_item_rating_training:
            for item in self.user_item_rating_training[user]:
                for rating in self.user_item_rating_training[user][item]:
                    if rating == "0":
                        squred_differences += (float(self.user_item_rating_test[user][item]) - self.training_results[item]) ** 2

        return sqrt(squred_differences / len(self.training_results))


'''
class Average(CollaborativeFilter):
    """Collaborative filter based on users' average ratings. By definition,
     is independent of user's identity.
    """

    training_data = None
    test_data = None    

    
    def readTrainingData(self, data):
        self.training_data = super(Average, self).readTrainingData(data)
        flat_list = ([x for x in [i for row in self.training_data for i in row]])
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
        flat_list = ([x for x in [i for row in self.test_data for i in row]])
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
'''       
        


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

class ItemData():
        item = None
        rating_to_user = {}
        

        def __init__(self, item, rating, user):
            print("init: ", item, rating, user)
            self.item = item
            self.rating = rating
            self.user = user

        pass


class ItemCosine(CollaborativeFilter):
    training_data = None
    cosine_similarities = {}

    




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

        item_ratings = []

        print("ITEM -- RATING -- USER")
        for item, rating, user in z:
            i_data = ItemData(item, rating, user)
            item_ratings.append(i_data)
            print("item", item, "rating", rating, "user       ", user)
            item_to_ratings[item].append([rating, user])

        for ir in item_ratings:
            print("aaahhh:", ir.item, ir.user, ir.rating)


        item_pairs_to_ratings = {}
        for item1 in item_to_ratings.items():
            for item2 in item_to_ratings.items():
                item_pairs_to_ratings[item1[0], item2[0]] = [None]
                print("here", item1[1])

                
                    

                # don't compare the same item
                if(item1[0] != item2[0]):
                    print("\n\nCOMPARING", item1[0], item2[0])



                    print("before remove: ", item1[1], item2[1])
                    '''
                    # remove 0 ratings
                    for rating in item1[1]:
                        if(int(rating[0]) == 0):
                            item1[1].remove(rating)
                    for rating in item2[1]:
                        if(int(rating[0]) == 0):
                            item2[1].remove(rating)
                    '''
                    # LEFT OFF HERE #

                    print("after removed:", item1[1], item2[1])

                    print("yoyo", item1[1], item2[1])

                    '''
                    for rating in item1[1]:
                        print("SADLKJDF", rating)
                        if rating[1] not in item1[1][1]:
                            print(rating, "not in 1")
                    '''
                    # only add items which have been rated by the same set of users
                    users_who_rated_item1 = []
                    for rating_and_user in item1[1]:
                        print("WATER", rating_and_user[0])
                        if(int(rating_and_user[0]) != 0):
                            users_who_rated_item1.append(rating_and_user[1])
                    users_who_rated_item2 = []
                    for rating_and_user in item2[1]:
                        if(int(rating_and_user[0]) != 0):
                            users_who_rated_item2.append(rating_and_user[1])

                    print("USERS WHO RATED ", item1[0],  users_who_rated_item1)
                    print("USERS WHO RATED ", item2[0],  users_who_rated_item2)

                    users_who_rated_both_items = set(users_who_rated_item1).intersection(users_who_rated_item2)

                    num_rated_item1_intersect_num_rated_both = set(users_who_rated_item1).intersection(users_who_rated_both_items)
                    num_rated_item2_intersect_num_rated_both = set(users_who_rated_item2).intersection(users_who_rated_both_items)

                    print("Users who rated both items: ", users_who_rated_both_items)
                    print("num1 in", num_rated_item1_intersect_num_rated_both)
                    print("num2 in", num_rated_item2_intersect_num_rated_both)

                    for rating_user in users_who_rated_item1:
                        print("SSSS", rating_user)
                        if(rating_user not in users_who_rated_both_items and len(users_who_rated_both_items) > 1):
                            users_who_rated_item1.remove(rating_user)

                    for rating_user in users_who_rated_item2:
                        print("SSSS", rating_user)
                        if(rating_user not in users_who_rated_both_items and len(users_who_rated_both_items) > 1):
                            users_who_rated_item2.remove(rating_user)

                    for i in item1[1]:
                        print("FAC", i)
                    for i in item2[1]:
                        print("FAC", i)

                    users_who_rated_item1_non_zero = []
                    for i in users_who_rated_item1:
                        users_who_rated_item1_non_zero.append(i)

                    for i in users_who_rated_item1_non_zero:
                        print("a", item1[0], i)

                    if len(num_rated_item1_intersect_num_rated_both) > 0 or len(num_rated_item2_intersect_num_rated_both) > 0:
                        print("ADDING", item1[0],":", users_who_rated_item1, item2[0], ":", users_who_rated_item2)
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
            print(item, item_and_user)
            if item[0] != None and item_and_user[0] != None and item[1] != None and item_and_user[1] != None:
                print("WWW:", item[0], item_and_user[0], item[1], item_and_user[1], "WWW")

                for rating_user in item_and_user[0]:
                    print("BOTTLE", rating_user)

                users_who_rated_item1_non_zero = []
                for i in item_and_user[0]:
                    print(i[0])
                    if(int(i[0]) != 0):
                        users_who_rated_item1_non_zero.append(i)

                for i in users_who_rated_item1_non_zero:
                    print("a", i)


                users_who_rated_item2_non_zero = []
                for i in item_and_user[1]:
                    print(i[0])
                    if(int(i[0]) != 0):
                        users_who_rated_item2_non_zero.append(i)

                for i in users_who_rated_item2_non_zero:
                    print("ab", i)

                users_who_rated_both_items_non_zero = []

                for rating in users_who_rated_item1_non_zero:
                    users_who_rated_both_items_non_zero.append(rating[1])

                for rating in users_who_rated_item2_non_zero:
                    users_who_rated_both_items_non_zero.append(rating[1])

                for user in users_who_rated_both_items_non_zero:
                    print("OCCU", user)

                users_who_rated_both_items_non_zero_common_to_both_items = []


                for user in users_who_rated_both_items_non_zero:
                    if users_who_rated_both_items_non_zero.count(user) > 1:
                        users_who_rated_both_items_non_zero_common_to_both_items.append(user)

                users_who_rated_both_items_non_zero_common_to_both_items = (set(users_who_rated_both_items_non_zero_common_to_both_items))

                print("MOUNTAIN:", users_who_rated_item1_non_zero)
                print("MOUNTAIN2:", users_who_rated_item2_non_zero)

                users_who_rated_item1_non_zero_and_is_common = []

                for rating_user in users_who_rated_item1_non_zero:
                    if int(rating_user[0]) != 0:
                        users_who_rated_item1_non_zero_and_is_common.append(rating_user[1])

                for i in users_who_rated_item1_non_zero_and_is_common:
                    print("JACK", i)


                users_who_rated_item2_non_zero_and_is_common = []

                for rating_user in users_who_rated_item2_non_zero:
                    if int(rating_user[0]) != 0:
                        users_who_rated_item2_non_zero_and_is_common.append(rating_user[1])

                for i in users_who_rated_item2_non_zero_and_is_common:
                    print("JACK2", i)



                rated_1 = []
                for rating in users_who_rated_item1_non_zero_and_is_common:
                    rated_1.append(rating[0])

                rated_2 = []
                for rating in users_who_rated_item2_non_zero_and_is_common:
                    rated_2.append(rating[0])

                print(rated_1)
                print(rated_2)
                rated_1_and_2 = set(rated_1).intersection(rated_2)

                for i in rated_1_and_2:
                    print(i)


                if str(3) in rated_1_and_2:
                    print("yes")
                else:
                    print("No")


                for user_rating in users_who_rated_item1_non_zero:
                    print("xx", user_rating[1])
                    if user_rating[1] not in rated_1_and_2:
                        print("removing", user_rating[1])
                        users_who_rated_item1_non_zero.remove(user_rating)

                for i in users_who_rated_item1_non_zero:
                    print(i)

                for user_rating in users_who_rated_item2_non_zero:
                    if user_rating[1] not in rated_1_and_2:
                        users_who_rated_item2_non_zero.remove(user_rating)

                for i in users_who_rated_item2_non_zero:
                    print(i)

                users_who_rated_item1_final = []


                for user in users_who_rated_item1_non_zero:
                    users_who_rated_item1_final.append(user[1])

                users_who_rated_item2_final = []


                for user in users_who_rated_item2_non_zero:
                    users_who_rated_item2_final.append(user[1])


                print("as", set(users_who_rated_item1_final))
                print("as1", set(users_who_rated_item2_final))



                final_intersection = set(users_who_rated_item1_final).intersection(users_who_rated_item2_final)

                print(final_intersection)

                final_1 = []
                for user in users_who_rated_item1_final:
                    if user in final_intersection:
                        print(user, "in ", final_intersection)
                        final_1.append(user)
                
                for user in final_1:
                    print('x', user)
                
                final_2 = []
                for user in users_who_rated_item2_final:
                    if user in final_intersection:
                        print(user, "in ", final_intersection)
                        final_2.append(user)
                
                for user in final_2:
                    print('y', user)

                print("v", users_who_rated_item1_non_zero)
                print("v2", users_who_rated_item2_non_zero)

                for rating_user in users_who_rated_item1_non_zero:
                    if rating_user[1] not in final_1:
                        users_who_rated_item1_non_zero.remove(rating_user)

                print("lsdkj", users_who_rated_item2_non_zero)

                for rating_user in users_who_rated_item2_non_zero:
                    print("why", rating_user[1])
                    if rating_user[1] not in final_2:
                        users_who_rated_item2_non_zero.remove(rating_user)

                print("w", users_who_rated_item1_non_zero)
                print("w2", users_who_rated_item2_non_zero)





                dot_prod = self.dotProduct(users_who_rated_item1_non_zero, users_who_rated_item2_non_zero)
                mag_vec0 = self.vecMagnitude(users_who_rated_item1_non_zero)
                mag_vec1 = self.vecMagnitude(users_who_rated_item2_non_zero)
                cosine_similarity = dot_prod/(mag_vec0 * mag_vec1)
                key_table = [item[0][0]]
                key_table.append(item[1][0])

                # convert to tuple so it can be hashed & used as dict key
                key_tuple = tuple(key_table)
                               
                self.cosine_similarities[key_tuple] = cosine_similarity

                if(DEBUG):
                    print("\nItems: ", item[0], item[1])
                    print("Dot product of ", users_who_rated_item1_non_zero[0], " and ", users_who_rated_item2_non_zero[0], " = ", dot_prod)
                    print("vec1 magnitude: ", mag_vec0)
                    print("vec2 magnitude: ", mag_vec1)
                    print("Cosine Similarity: ", cosine_similarity)

    def dotProduct(self, vec1, vec2):

        vec1_ratings = []
        for i in vec1:
            vec1_ratings.append(i[1])

        vec2_ratings = []
        for i in vec2:
            vec2_ratings.append(i[1])

        intersec = set(vec2_ratings).intersection(vec1_ratings)

        for rating in vec1:
            if(rating[1] not in intersec):
                vec1.remove(rating)

        for rating in vec2:
            if(rating[1] not in intersec):
                vec2.remove(rating)




        print("sl;kdfj;slajf;lasjf;lskdjf", vec1)
        print("sl;kdfj2222222lskdjf", vec2)




        print("IN DOTPRODCUT(): ", vec1, vec2)
        if len(vec1) != len(vec2):
            print("ERROR - vector length mismatch")
        
        else:
            # extract ratings from vec1, vec2 with index 0 - [rating, user][rating, user]...[rating, user]
            v1 = []
            for rating_user in vec1:
                v1.append(rating_user[0])

            v2 = []
            for rating_user in vec2:
                v2.append(rating_user[0])

            print("CALCULATING DOT PRODUCT OF: ", v1, v2)

            z = zip(v1, v2)
            dot_prod = 0
            for rating in z:
                dot_prod += int(rating[0]) * int(rating[1]) 
            print("DOT PRODUCT OF: ", v1, v2, " = ", dot_prod)
            return dot_prod    

    def vecMagnitude(self, vec):
        # extract ratings from vec with index 0 - [rating, user][rating, user]...[rating, user]
        v = []
        for rating_user in vec:
            v.append(rating_user[0])

        print("CALCULATING MAGNITUDE OF ", v)

       
        mag = 0
        for i in v:
            mag += int(i) ** 2
        print("MAGNITUDE OF ", v, " = ", sqrt(mag))
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
    training_data = None
    test_data = None

    user_item_rating_training = None
    user_item_rating_test = None

    training_results = None
    test_results = None

    def readTrainingData(self, data):
        self.training_data = super(SlopeOne, self).readTrainingData(data)

        user_col = 0
        item_col = 1
        rating_col = 2
        timestamp_col = 3

        items = [row[item_col] for row in self.training_data] 
        ratings = [row[rating_col] for row in self.training_data] 
        users = [row[user_col] for row in self.training_data] 

        z = zip(items, users, ratings)

        item_user_rating = UserItemRating()
        for item, user, rating in z:
            item_user_rating[item][user] =  rating

        print(item_user_rating)

      
        for item1 in item_user_rating:
            
            for item2 in item_user_rating:
                dif = 0
                num_ratings = 0
                
                print(item1, item2)
                for user1 in item_user_rating[item1]:
                    for user2 in item_user_rating[item2]:
                        if user1 == user2:
                            if item1 != item2:
                                
                                for rating in item_user_rating[item1][user1]:
                                    print(item1, item2, item_user_rating[item1][user1], item_user_rating[item2][user2])
                                    if item_user_rating[item1][user1] != "0" and item_user_rating[item2][user2] != "0":
                                        num_ratings += 1
                                        dif += (float(item_user_rating[item2][user2]) - float(item_user_rating[item1][user1]))
                if num_ratings != 0:
                    print("end", num_ratings, dif / int(num_ratings))               
                print("DIF:", dif)




    def readTestData(self, data):
        pass
    

    def itemDifferences(self):
        pass
    

        '''
        for user1 in self.user_item_rating_training:
            for user2 in self.user_item_rating_training:
                count = 0
                dif = 0
                for item1 in self.user_item_rating_training[user1]:
                    for item2 in self.user_item_rating_training[user2]:
                        if user1 != user2 and item1 != item2:
                            for rating1 in self.user_item_rating_training[user1][item1]:
                                for rating2 in self.user_item_rating_training[user2][item2]:
                                    if rating1 != "0" and rating2 != "0":
                                        count += 1
                                        dif += float(rating2) - float(rating1)
                                        print(user1, user2, item1, rating1, item2, rating2, dif)
                print("c", count)

        for item1 in item_pair_diffs:
            for item2 in item_pair_diffs:
                print(item1, item2, item_pair_diffs[item1][item2])
        '''
    def RMSE(self):
        pass
        '''
        squred_differences = 0
        rating_diffs = []
        for user in self.user_item_rating_training:
            for item in self.user_item_rating_training[user]:
                for rating in self.user_item_rating_training[user][item]:
                    if rating == "0":
                        squred_differences += (float(self.user_item_rating_test[user][item]) - self.training_results[item]) ** 2

        return sqrt(squred_differences / len(self.training_results))
        '''


'''
class SlopeOne(CollaborativeFilter):
    training_data = None

    item_list = []
    item_dict = {}
    item_matrix = {}
    user_items_ratings = {}
    user_list = []
    user_item_rating = None
    item_set = None
    user2prediction = []
    uir_predictions = None
    item_pair_list = []


    def readTrainingData(self, data):

        self.training_data = super(SlopeOne, self).readTrainingData(data)

        
        user_col = 0
        item_col = 1
        rating_col = 2
        timestamp_col = 3

        items = [row[item_col] for row in self.training_data] 
        ratings = [row[rating_col] for row in self.training_data] 
        users = [row[user_col] for row in self.training_data] 
        self.user_list = set(users)


        item_to_ratings = defaultdict(list)

        z = zip(items, ratings, users)


        item_list = []

        for item, rating, user in z:
            self.item_list.append(item)

        for item, rating, user in z:
            print(item, rating)

        item_set = set(self.item_list)

        item_set = [item for item in item_set]

        for i in item_set:
            item_to_ratings[i] = i

      
        z = zip(items, ratings, users)

        self.item_dict = defaultdict(list)
        for item, rating, user in z:
            for i in item_to_ratings:
                if item == i:
                    self.item_dict[item].append(rating)

        z = zip(users, items, ratings)

        self.user_item_rating = UserItemRating()
        for user, item, rating in z:
            print(user, item, rating)
            self.user_item_rating[user][item] =  rating

        for i in range(0, 100):
            print("\n")

        print(self.user_item_rating)

        for i in range(0, 100):
            print("\n")

        user = {}
        for i in self.item_dict.items():
            # print("z", i[1], end="\n")
            pass

       
       

        for item1 in self.item_dict.items():
            for item2 in self.item_dict.items():
                if item1 != item2:
                    # print("item1 ", item1[0], item1[1], "item 2", item2[0], item2[1])
                    diff_per_rating = [int(rating_1) - int(rating_2) for rating_1, rating_2 in zip(item1[1], item2[1]) if int(rating_1) != 0 and int(rating_2) != 0]


                    # print("diff", item1[0], item2[0], diff_per_rating)
                    diff = 0

                    for i in diff_per_rating:
                        diff += i
                    # print("here", diff)
                    items = [item1[0]]
                    items.append(item2[0])
                    items = tuple(items)
                    print(item1, item2, diff, "\n")
                    # self.item_matrix[items] = diff / len(set(diff_per_rating))
                    difference = diff / len(set(diff_per_rating))
                    # self.item_matrix[1] = len(set(diff_per_rating))
                    k = []
                    k.append(difference)
                    k.append(len(set(diff_per_rating)))
                    k = tuple(k)
                
                    self.item_matrix[items] = k

        # self.item_matrix = sorted(self.item_matrix)



        items = [row[item_col] for row in self.training_data] 
        ratings = [row[rating_col] for row in self.training_data] 
        users = [row[user_col] for row in self.training_data] 

        self.item_set = set(self.item_list)
        

        self.uir_predictions = UserItemRating()
        for user in self.user_list:
            for item in self.item_list:
                if self.user_item_rating[user][item] == "0":
                    self.makePrediction(user, item)
              



    def makePrediction(self, user, unrated_item):
        print("\n\n\n\n\n\nMAKING PRED FOR ", user, unrated_item)
        for user in self.user_list:
            for item in self.item_set:
                    user_rating = self.user_item_rating[user][item]
                    user_item_rating_pertinent = item

                    for item_pair in self.item_matrix:                      

                            item_user_has_rated = None
                            for i in self.user_item_rating[user]:
                                if i in item_pair:
                                    if self.user_item_rating[user][i] != "0":
                                        item_user_has_rated = i
                                        print(i, "is in ", item_pair)




                            self.item_pair_list.append(item_pair)
                            print("acorn", item_pair[0])

                            print("based on ", item_pair, " he will give it a ", self.item_matrix[item_pair])

                            print("\n\n\n PHONE\n", user, item, "\n\n", self.user_item_rating[user][user_item_rating_pertinent], "\n\n\n")
                            if(len(self.user_item_rating[user][item_user_has_rated]) > 0):
                                self.uir_predictions[user][item][item_pair] = self.item_matrix[item_pair][0] + int(str(self.user_item_rating[user][item_user_has_rated]))
                            print("USER", user, "ITEM PAIR", item_pair, "UNRATED ITEM", unrated_item, "PREDICTION",  self.uir_predictions[user][item][item_pair])

        print(self.uir_predictions)

        pass

    def showPredictions(self):

        user_set = set(self.user_list)
        item_set = set(self.item_list)
        item_pair_set = set(self.item_pair_list)


        for user in user_set:
            for item in item_set:
                for item_pair in item_pair_set:
                    if self.user_item_rating[user][item] == "0" and item == item_pair[0]:
                        print("USER", user, "WILL RATE", item, "A SCORE OF ", self.uir_predictions[user][item][item_pair], "BASED ON", item_pair[0], "using", item_pair[1])

        print(self.user_item_rating)

        
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

    # rmse = sqrt(sum_squared_dif/num_obvs)
    return 3              
        
             
    pass
'''