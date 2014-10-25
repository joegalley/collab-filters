import sys
import time

from collaborativeFilters.filters import (
    Average,
    UserEucledian,
    UserPearson,
    ItemCosine,
    ItemAdjustedCosine,
    SlopeOne
)


DEBUG = True

def main():
    if(len(sys.argv) != 4):
        print("ERROR - usage: {0} <training.data> <test.data> <algorithm>".format(sys.argv[0]))
        sys.exit(2)

    training_data_file = sys.argv[1]
    test_data_file = sys.argv[2]
    collab_algo = sys.argv[3]
    rmse = 2

    start = time.time()
    if(collab_algo == "average"):    
        avg = Average()
        avg.readTrainingData(training_data_file)
        avg.readTestData(test_data_file)
        rmse = avg.RMSE()
    elif(collab_algo == "user-eucledian"):    
        user_euc = UserEucledian()
        user_euc.readTrainingData(training_data_file)
    elif(collab_algo == "item-cosine"):    
        item_cosine = ItemCosine()
        item_cosine.readTrainingData(training_data_file)
        item_cosine.showCosineSimilarities()
    elif(collab_algo == "item-adcosine"):    
        item_adcosine = ItemAdjustedCosine()
        item_adcosine.readTrainingData(training_data_file)
        item_adcosine.getUsersAvgRatings(training_data_file)
        item_adcosine.showCosineSimilarities()
    elif(collab_algo == "user-pearson"):    
        pearson = UserPearson()
        pearson.readTrainingData(training_data_file)
        
    elif(collab_algo == "slope-one"):    
        slope_one = SlopeOne()
        slope_one.readTrainingData(training_data_file)
        # slope_one.readTestData(test_data_file)
        slope_one.itemDifferences()
        rmse = slope_one.RMSE()
    end = time.time()

    print("\n\nMYRESULTS Training = ", training_data_file, sep="")
    print("MYRESULTS Testing = ", test_data_file, sep="")
    print("MYRESULTS Algorithm = ", collab_algo, sep="")
    print("MYRESULTS RMSE = ", rmse, sep="")
    if DEBUG:
        print("TIME ELAPSED: ", "{:.2f}".format(end - start),"s")

    

if __name__ == "__main__":
    main()