# report
Simple python module to generate bokeh plots from (almost) arbitrary pandas data frames.

# Example usage:

```python
import report
from bokeh.palettes import Spectral6

groups = [
    {'sample': 'sample_A', 'augmentation':0},
    {'sample': 'sample_B', 'augmentation':0},
    {'sample': 'sample_C', 'augmentation':0},
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

# read results into a pandas data frame
results = report.read_all_results()

report.plot(groups, figures, configurations, results)
```
