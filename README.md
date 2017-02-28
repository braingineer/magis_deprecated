# Models and Algorithms for Grounded, Interactive Semantics

original post: [mcmahan.io/lux](http://mcmahan.io/lux)

Since the overhaul:

1. new version of lux; not published
2. dataset interfaces

run the `pip install -r requirements.txt`. It includes a reference to `eidos`, my repo for convenience constructs. 

important note about colors like red:
 - color space is on hsv, and hue is a circle
 - rather than use von mises, we used a space transformation
 - for all colors, their hue domain is either 0,2pi or -pi,pi. 
 - we transform their data to both, see which has the smaller variance, and leave it there. 
 - so if you see negative hue numbers, that's why 

check out [the example notebook](https://github.com/braingineer/magis/blob/master/public-magis-example.ipynb)

this is the same as the notebook:

```python

import magis

lux = magis.models.Lux.pretrained()

#### lux.components has all of the individual color models
#### there's no shortcut for the names right now because I just overhauled and hadn't needed it yet

color_names = {c.name:c for c in lux.components}

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
