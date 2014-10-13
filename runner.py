import sys

from collaborativeFilters.filters import (
	Average,
	UserEucledian,
	UserPearson,
	ItemCosine,
	ItemAdjustedCosine,
	SlopeOne
)



def main():
	if(len(sys.argv) != 4):
	    print("ERROR - usage: {0} <training.data> <test.data> <algorithm>".format(sys.argv[0]))
	    sys.exit(2)

	training_data_file = sys.argv[1]
	test_data_file = sys.argv[2]
	collab_algo_file = sys.argv[3]
	rmse = 2


	avg = Average()
	avg.readTrainingData(training_data_file)
	avg.readTestData(test_data_file)



	print("\n\nRESULTS Training = ", training_data_file, sep="")
	print("RESULTS Testing = ", test_data_file, sep="")
	print("RESULTS Algorithm = ", collab_algo_file, sep="")
	print("RESULTS RMSE = ", rmse, sep="")



if __name__ == "__main__":
	main()