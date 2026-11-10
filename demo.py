import sys
import time

from collaborativeFilters.filters import Average, SlopeOne

ALGORITHMS = {
    "average": Average,
    "slope-one": SlopeOne,
}


def main():
    if len(sys.argv) != 4:
        print("usage: {0} <training.tsv> <test.tsv> <algorithm>".format(sys.argv[0]))
        print("algorithms: {0}".format(", ".join(ALGORITHMS)))
        sys.exit(2)

    training_file, test_file, algorithm = sys.argv[1:4]

    filter_cls = ALGORITHMS.get(algorithm)
    if filter_cls is None:
        print("unknown algorithm '{0}'. choose from: {1}".format(algorithm, ", ".join(ALGORITHMS)))
        sys.exit(2)

    start = time.time()
    collab_filter = filter_cls().fit(training_file)
    rmse = collab_filter.rmse(test_file)
    elapsed = time.time() - start

    print("training file: {0}".format(training_file))
    print("test file:     {0}".format(test_file))
    print("algorithm:     {0}".format(algorithm))
    print("RMSE:          {0:.4f}".format(rmse))
    print("elapsed:       {0:.2f}s".format(elapsed))


if __name__ == "__main__":
    main()
