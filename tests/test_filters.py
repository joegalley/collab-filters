from pathlib import Path

import pytest

from collaborativeFilters.filters import Average, CollaborativeFilter, SlopeOne

EXAMPLES_DIR = Path(__file__).resolve().parent.parent / "examples"


def write_ratings(path, rows):
    path.write_text("\n".join("\t".join(str(field) for field in row) for row in rows) + "\n")


def test_collaborative_filter_cannot_be_instantiated_directly():
    with pytest.raises(TypeError):
        CollaborativeFilter()


def test_average_predicts_item_mean_for_held_out_ratings(tmp_path):
    training = tmp_path / "training.tsv"
    test = tmp_path / "test.tsv"

    write_ratings(
        training,
        [
            ("u1", "i1", 4, 0),
            ("u1", "i2", 0, 0),  # held out
            ("u2", "i1", 2, 0),
            ("u2", "i2", 5, 0),
        ],
    )
    write_ratings(test, [("u1", "i2", 3, 0)])

    avg = Average().fit(str(training))

    assert avg.predictions == {("u1", "i2"): 5.0}
    assert avg.rmse(str(test)) == pytest.approx(2.0)


def test_slope_one_predicts_from_pairwise_item_deviation(tmp_path):
    training = tmp_path / "training.tsv"
    test = tmp_path / "test.tsv"

    write_ratings(
        training,
        [
            ("u1", "i1", 5, 0),
            ("u1", "i2", 3, 0),
            ("u2", "i1", 4, 0),
            ("u2", "i2", 2, 0),
            ("u3", "i1", 5, 0),
            ("u3", "i2", 0, 0),  # held out
        ],
    )
    write_ratings(test, [("u3", "i2", 3.5, 0)])

    slope_one = SlopeOne().fit(str(training))

    assert slope_one.predictions[("u3", "i2")] == pytest.approx(3.0)
    assert slope_one.rmse(str(test)) == pytest.approx(0.5)


@pytest.mark.parametrize("filter_cls", [Average, SlopeOne])
def test_rmse_before_fit_raises(filter_cls, tmp_path):
    test = tmp_path / "test.tsv"
    write_ratings(test, [("u1", "i1", 3, 0)])

    with pytest.raises(RuntimeError):
        filter_cls().rmse(str(test))


@pytest.mark.parametrize("filter_cls", [Average, SlopeOne])
def test_example_dataset_end_to_end(filter_cls):
    collab_filter = filter_cls().fit(str(EXAMPLES_DIR / "training.tsv"))

    assert collab_filter.predictions  # some pairs were held out and predicted

    rmse = collab_filter.rmse(str(EXAMPLES_DIR / "test.tsv"))
    assert rmse >= 0
