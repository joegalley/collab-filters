# collab-filters

A python module for various [collaborative filtering](http://en.wikipedia.org/wiki/Collaborative_filtering) algorithms.


### Including in your project:

```
from collaborativeFilters.filters import (
    Average,
    UserEucledian,
    UserPearson,
    ItemCosine,
    ItemAdjustedCosine,
    SlopeOne,
    RSME
)
```

You can also run this as a stand-alone script to get collaborative filter infomration from your data.


### Usage:
```
> python demo.py <training_data>.txt <test_data>.txt <algorithm>
```


View a demo in [```demo.py```](https://github.com/sponde210/collab-filters/blob/master/demo.py)

#### This project is still in development.