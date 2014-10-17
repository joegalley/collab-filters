import sys

from collaborativeFilters.filters import (
    Average,
    UserEucledian,
    UserPearson,
    ItemCosine,
    ItemAdjustedCosine,
    SlopeOne,
    RSME
)



def main():
    if(len(sys.argv) != 4):
        print("ERROR - usage: {0} <training.data> <test.data> <algorithm>".format(sys.argv[0]))
        sys.exit(2)

    training_data_file = sys.argv[1]
    test_data_file = sys.argv[2]
    collab_algo = sys.argv[3]
    rmse = 2

    if(collab_algo == "average"):    
        avg = Average()
        avg.readTrainingData(training_data_file)
        avg.readTestData(test_data_file)
        rmse = avg.calculateError()
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

    print("\n\nRESULTS Training = ", training_data_file, sep="")
    print("RESULTS Testing = ", test_data_file, sep="")
    print("RESULTS Algorithm = ", collab_algo, sep="")
    print("RESULTS RMSE = ", rmse, sep="")

    

if __name__ == "__main__":
    main()