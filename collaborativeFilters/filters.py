"""Collaborative filtering algorithms: item averages and weighted Slope One.

Ratings files are tab-separated rows of ``user, item, rating, timestamp``.
A rating of ``0`` in a training file marks a (user, item) pair that has been
held out for evaluation; the true rating for that pair belongs in the test
file instead, so ``rmse()`` can score the filter's predictions against it.
"""

from abc import ABC, abstractmethod
from collections import defaultdict
from math import sqrt


def _read_ratings(path):
    rows = []
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            user, item, rating, timestamp = line.split("\t")
            rows.append((user, item, float(rating), timestamp))
    return rows


class CollaborativeFilter(ABC):
    """Base class for collaborative filters.

    Subclasses implement ``fit()``, which reads a training file and
    populates ``self.predictions`` as a ``{(user, item): predicted_rating}``
    mapping for every pair held out in that file.
    """

    def __init__(self):
        self.predictions = {}

    @abstractmethod
    def fit(self, training_file):
        """Read training_file and compute predictions for held-out pairs."""

    def rmse(self, test_file):
        """Root-mean-squared error of self.predictions against test_file."""
        if not self.predictions:
            raise RuntimeError("fit() must be called before rmse()")

        truth = {(user, item): rating for user, item, rating, _ in _read_ratings(test_file)}
        squared_errors = [
            (truth[pair] - predicted) ** 2
            for pair, predicted in self.predictions.items()
            if pair in truth
        ]
        if not squared_errors:
            raise ValueError("no held-out pairs from training_file were found in test_file")

        return sqrt(sum(squared_errors) / len(squared_errors))


class Average(CollaborativeFilter):
    """Predicts each held-out rating as the mean of that item's known ratings."""

    def fit(self, training_file):
        item_ratings = defaultdict(list)
        held_out = []
        for user, item, rating, _ in _read_ratings(training_file):
            if rating == 0:
                held_out.append((user, item))
            else:
                item_ratings[item].append(rating)

        item_avg = {item: sum(ratings) / len(ratings) for item, ratings in item_ratings.items()}

        self.predictions = {
            (user, item): item_avg[item] for user, item in held_out if item in item_avg
        }
        return self


class SlopeOne(CollaborativeFilter):
    """Weighted Slope One.

    For every pair of items, averages how much users who rated both of them
    rated one above the other. A held-out rating is then predicted from the
    user's other ratings, each adjusted by that pairwise average and weighted
    by how many users supported it.
    """

    def fit(self, training_file):
        user_ratings = defaultdict(dict)
        held_out = []
        for user, item, rating, _ in _read_ratings(training_file):
            if rating == 0:
                held_out.append((user, item))
            else:
                user_ratings[user][item] = rating

        diff_sum = defaultdict(lambda: defaultdict(float))
        diff_count = defaultdict(lambda: defaultdict(int))
        for ratings in user_ratings.values():
            for item_i, rating_i in ratings.items():
                for item_j, rating_j in ratings.items():
                    if item_i == item_j:
                        continue
                    diff_sum[item_i][item_j] += rating_i - rating_j
                    diff_count[item_i][item_j] += 1

        deviation = {
            item_i: {item_j: diff_sum[item_i][item_j] / count for item_j, count in counts.items()}
            for item_i, counts in diff_count.items()
        }

        self.predictions = {}
        for user, item in held_out:
            numerator = 0.0
            denominator = 0
            for other_item, other_rating in user_ratings.get(user, {}).items():
                count = diff_count[item].get(other_item, 0)
                if count == 0:
                    continue
                numerator += (deviation[item][other_item] + other_rating) * count
                denominator += count
            if denominator:
                self.predictions[(user, item)] = numerator / denominator

        return self
