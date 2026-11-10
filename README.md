# collab-filters

A small Python implementation of two [collaborative filtering](https://en.wikipedia.org/wiki/Collaborative_filtering)
algorithms for predicting ratings:

- **Average** — predicts a user's rating for an item as that item's mean rating across all other users.
- **Slope One** — predicts a rating from the user's other ratings, each adjusted by the average
  rating difference between that item and the one being predicted, weighted by how many users
  support that comparison ([Lemire & Maclachlan, 2005](https://arxiv.org/abs/cs/0702144)).

## Data format

Both algorithms read tab-separated ratings files with one row per observation:

```
user	item	rating	timestamp
```

A rating of `0` in the **training** file marks a `(user, item)` pair that has been held out for
evaluation — the filter predicts a value for it instead of using it as input. The true rating for
that pair belongs in the **test** file, so `rmse()` can score the prediction against it.

See [`examples/training.tsv`](examples/training.tsv) and [`examples/test.tsv`](examples/test.tsv)
for a small worked example.

## Usage

As a library:

```python
from collaborativeFilters.filters import Average, SlopeOne

model = SlopeOne().fit("examples/training.tsv")
print(model.predictions)          # {(user, item): predicted_rating, ...}
print(model.rmse("examples/test.tsv"))
```

As a script:

```
python demo.py examples/training.tsv examples/test.tsv slope-one
```

`<algorithm>` is one of `average` or `slope-one`.

## Development

```
pip install -e ".[dev]"
pytest
```

## License

[MIT](LICENSE)
