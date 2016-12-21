# report
Simple python module to generate bokeh plots from (almost) arbitrary pandas data frames.

# Install

```
python setup.py install
```

# Example usage:

The following example will create a grid of plots. The rows are called
/groups/. Each group contains several /figures/. Each figure contains a plot
according to the given /configurations/.

/groups/ and /configurations/ are lists of dictionaries. Each dictionary
describes a filter on the data. Only rows in the `pandas` data frame that match
all filters are shown. If the value of a filter entry is a list, any value in
the list matches.

/figures/ is a dictionary with the mandatory keys `x_axis` and `y_axis`. The
values should have the names of columns with numerical data in the `pandas`
data frame. A figure can have an optional `title`.

/configurations/ can have optional keys `label` and `color` (which are not used
for matching).

```python
import report
from bokeh.palettes import Spectral6

# create a pandas data frame, you might want to replace that with your own way of reading results
results = report.read_all_results()

groups = [
    {'sample': 'sample_A', 'augmentation': 0},
    {'sample': 'sample_B', 'augmentation': 0},
    {'sample': 'sample_C', 'augmentation': 0},
]

figures = [
    {'x_axis': 'voi_split',  'y_axis': 'voi_merge',   'title': 'VOI'},
    {'x_axis': 'threshold',  'y_axis': 'voi_sum',     'title': 'VOI'},    
    {'x_axis': 'iteration',  'y_axis': 'cremi_score', 'title': 'CREMI score'},
    {'x_axis': 'rand_split', 'y_axis': 'rand_merge',  'title': 'RAND'},
]

configurations = [
    { 'setup':'setup35', 'iteration': 200000, 'tag': 'waterz', 'label': 'old caffe', 'color': Spectral6[0]},
    { 'setup':['setup%02d'%s for s in range(47,58)], 'tag': 'waterz', 'label': 'new caffe', 'color': Spectral6[1]},

]

report.plot(groups, figures, configurations, results)
```
