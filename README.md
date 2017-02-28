# Models and Algorithms for Grounded, Interactive Semantics


Since the overhaul:

1. new version of lux; not published
2. dataset interfaces

run the `pip install -r requirements.txt`. It includes a reference to `eidos`, my repo for convenience constructs. 

use:

```python

import magis

lux = magis.models.Lux.pretrained()

#### lux.components has all of the individual color models
#### there's no shortcut for the names right now because I just overhauled and hadn't needed it yet

color_names = {c.name:c for c in lux.componenets}

print(list(color_names.keys()))


### for the xkcd data:

xk = magis.data.XKCD()
xk.load()

### actual data is a bit buried b/c again.. just revamped. but, 

data_generator = xk.generate_forever('train')
print(next(data_generator))
### usually, used with itertools.islice

### actual data is down in xk._active_df

```
