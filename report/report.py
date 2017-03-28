import fnmatch
import os
import time
import json
import pandas
import numpy as np
import bokeh.plotting
import bokeh.layouts
import bokeh.models
import bokeh.palettes

verbose = False

configuration_keywords = ['color', 'label', 'style', 'title']

colors_rgb = [
[1, 0, 103],
[213, 255, 0],
[255, 0, 86],
[158, 0, 142],
[14, 76, 161],
[255, 229, 2],
[0, 95, 57],
[0, 255, 0],
[149, 0, 58],
[255, 147, 126],
[164, 36, 0],
[0, 21, 68],
[145, 208, 203],
[98, 14, 0],
[107, 104, 130],
[0, 0, 255],
[0, 125, 181],
[106, 130, 108],
[0, 174, 126],
[194, 140, 159],
[190, 153, 112],
[0, 143, 156],
[95, 173, 78],
[255, 0, 0],
[255, 0, 246],
[255, 2, 157],
[104, 61, 59],
[255, 116, 163],
[150, 138, 232],
[152, 255, 82],
[167, 87, 64],
[1, 255, 254],
[255, 238, 232],
[254, 137, 0],
[189, 198, 255],
[1, 208, 255],
[187, 136, 0],
[117, 68, 177],
[165, 255, 210],
[255, 166, 254],
[119, 77, 0],
[122, 71, 130],
[38, 52, 0],
[0, 71, 84],
[67, 0, 44],
[181, 0, 255],
[255, 177, 103],
[255, 219, 102],
[144, 251, 146],
[126, 45, 210],
[189, 211, 147],
[229, 111, 254],
[222, 255, 116],
[0, 255, 120],
[0, 155, 255],
[0, 100, 1],
[0, 118, 255],
[133, 169, 0],
[0, 185, 23],
[120, 130, 49],
[0, 255, 198],
[255, 110, 65],
[232, 94, 190]
]
colors = [ '#%02x%02x%02x'%tuple(c) for c in colors_rgb ]

# decorator for filter configurations that allow multiple values
class Any:
    def __init__(self, values):
        self.values = values

def fetch_data_frame_from_json(data_dir='processed'):

    start = time.time()

    print("Collecting record files...")
    record_files = []
    for root, _, filenames in os.walk(data_dir):
        for filename in fnmatch.filter(filenames, '*.json'):
                record_files.append(os.path.join(root, filename))
    if verbose:
        print("Found record files:")
        for f in record_files:
            print('\t' + f)

    print("Parsing record files...")
    all_records = []
    for f in record_files:
        try:
            all_records.append(json.load(open(f, 'r')))
        except:
            print("record file " + f + " seems broken (maybe in process?)")
            pass

    print("Creating data frame...")
    columns = set()
    for r in all_records:
        for k in r:
            columns.add(k)

    data_frame = pandas.DataFrame(index=range(len(all_records)), columns=list(columns))
    for i in range(len(all_records)):
        data_frame.loc[i] = all_records[i]

    print("Read all results in " + str(time.time() - start) + "s")

    return data_frame

def read_all_results(data_backend = 'json_files', **kwargs):
    '''Reads all results provided by the given data backend, and returns them
    as a panda data frame with at least the following columns:

        voi_split, voi_merge, voi_sum, rand_split, rand_merge, arand, cremi_score

    More columns might be present to describe the configuration that produced
    the results. There are no assumptions about the number, names, or datatypes
    of these configuration columns. Their presence depends entirely on the data
    backend from where the results are read.

    The columns

        voi_sum, arand, cremi_score

    will be computed from {voi,rand}_{split,merge} and should not be provided
    by the data backend.

    Available data backends:

        'json_files'

            Parses all '*.json' files under the 'processed' directory
            (including subdirectories). Each file is supposed to contain one
            row of the data frame as a dictionary, with column names as
            dictionary keys.
    '''

    if data_backend == 'json_files':
        data_frame = fetch_data_frame_from_json(**kwargs)
    else:
        raise RuntimeError("no such data backend '" + str(data_backend) + "'")

    try:
        ts = data_frame['ted_split']
        tm = data_frame['ted_merge']
        tp = data_frame['ted_fp']
        tn = data_frame['ted_fn']
        data_frame['ted_sum'] = np.array(ts+tm+tp+tn, dtype=np.float32)
    except:
        pass

    try:
        vs = data_frame['voi_split']
        vm = data_frame['voi_merge']
        data_frame['voi_sum'] = np.array(vs + vm, dtype=np.float32)
    except:
        pass

    try:
        rs = data_frame['rand_split']
        rm = data_frame['rand_merge']
        data_frame['arand'] = np.array(1.0 - (2.0*rs*rm)/(rs+rm), dtype=np.float32)
        data_frame['cremi_score'] = np.sqrt(np.array(data_frame['voi_sum']*data_frame['arand'], dtype=np.float32))
    except:
        pass

    return data_frame

def matches(record, configuration):

    # symbolic group 'average' matches all
    if isinstance(configuration, str) and configuration == 'average':
        return True

    for key in configuration.keys():

        # skip non-record fields
        if key in configuration_keywords:
            continue

        if key not in record:
            return False

        allowed_values = configuration[key]
        if not isinstance(allowed_values, list):
            allowed_values = [allowed_values]

        if record[key] not in allowed_values:
            return False

    return True

def key_value_to_str(d, k):
    if isinstance(d[k], basestring):
        return d[k]
    return str(k)[:2] + ":" + str(d[k])

def get_configuration_label(configuration):

    if 'label' in configuration:
        return configuration['label']

    label = ""

    keys = sorted(list(configuration.keys()))
    keys = [ k for k in keys if not k in configuration_keywords ]

    return ", ".join([key_value_to_str(configuration, k) for k in keys])

def get_title(group):

    if 'title' in group:
        return str(group['title'])
    return str(group)

def filter(records, configurations):

    match_mask = None

    values_queries = []
    for c in configurations:
        for k in c:

            if k in configuration_keywords:
                continue

            possible_values = c[k].values if isinstance(c[k], Any) else [c[k]]

            value_mask = None

            for value in possible_values:
                if value is None or value is np.nan:
                    expression = (pandas.isnull(records[k]))
                else:
                    expression = (records[k] == value)
                if value_mask is None:
                    value_mask = expression
                else:
                    value_mask |= expression

            values_queries.append("(" + " or ".join([str(k) + " == " + str(v) for v in possible_values]) + ")")

            if match_mask is None:
                match_mask = value_mask
            else:
                match_mask &= value_mask

    if len(values_queries) == 0:
        return records

    query_string = " and ".join(values_queries)

    if verbose:
        print("Filtering with: " + query_string)

    filtered = records[match_mask]
    return filtered

def smooth(keys, values, factor):

    assert(len(keys) == len(values))
    if len(keys) == 0:
        return keys, values

    # co-sort keys and values by keys
    keys, values = map(np.array, zip(*sorted(zip(keys, values))))

    values_mean = np.mean(
            np.array(
                values[:int(np.floor(len(values)/factor))*factor]
            ).reshape(-1,factor),
            axis=1
    )
    keys_mean = np.mean(
            np.array(
                keys[:int(np.floor(len(keys)/factor))*factor]
            ).reshape(-1,factor),
            axis=1
    )

    return keys_mean, values_mean

def plot(groups, figures, configurations, all_records):
    '''Creates figures from the given records.

    Figures are created for each given group.

    Curves are drawn in the figures for each given configuration.
    '''

    # configurations are dictionaries with keys like
    #
    #  sample, setup, iteration, threshold
    #
    # (basically everything that can occur in a record)
    #
    # Values are one or more values, e.g.,
    #
    #   'setup': 'setup26',
    #   'threshold': [0,100]
    #
    # We create a curve for every configuration out of all records that have the 
    # given keys set to the same value(s).

    print("Preparing plot data")
    start = time.time()

    # list of curves per group, identified by title
    curves = { get_title(group): [] for group in groups }

    # prepare panda data frames for each curve in each group
    for group in groups:

        if verbose:
            print("processing group " + str(group))

        configuration_num = -1
        for configuration in configurations:
            configuration_num += 1

            if verbose:
                print("adding configuration " + str(configuration))

            filtered_records = filter(all_records, [configuration, group])

            if verbose:
                print("filtered records for " + str(configuration) + ", group " + str(group) + ":")
                print(filtered_records)

            # create curve with meta-data
            curve = {
                'columns': filtered_records,
                'label': get_configuration_label(configuration),
                'color': colors[configuration_num%len(colors)] if 'color' not in configuration else configuration['color'],
                'style': configuration['style'] if 'style' in configuration else 'circle'
            }

            curves[get_title(group)].append(curve)

    print("Prepared data in " + str(time.time() - start) + "s")

    # plotting
    start = time.time()

    average_curve = {}

    keys = list(all_records.keys())
    keys.sort()

    # create the figures for each group
    for group in groups:

        if isinstance(group, str) and group == 'average':
            group_curves = [ v for (_,v) in average_curve.iteritems() ]
        else:
            group_curves = curves[get_title(group)]

        if len(group_curves) == 0:
            print("No record matches group " + get_title(group))
            continue

        group_figures = []
        for figure in figures:

            # configure the tool-tip to show all keys
            tooltips="".join([
                "<div><span>%s: @%s</span></div>"%(key,key) for key in keys
            ])
            hover = bokeh.models.HoverTool(tooltips=tooltips)
            tools = ['save','pan','wheel_zoom','box_zoom','reset']

            show_legend = len(group_curves) < 5

            # create the bokeh figure
            group_figure = bokeh.plotting.figure(
                    title=get_title(group) + " " + get_title(figure),
                    tools=[hover] + tools,
                    active_scroll='wheel_zoom',
                    x_axis_label=figure['x_axis'],
                    y_axis_label=figure['y_axis'])

            for curve in group_curves:

                columns = curve['columns']

                # bokeh does not handle nan in python notebooks correctly, we 
                # replace them with a string here
                columns = columns.fillna('nan')

                if 'smooth' in figure and figure['smooth'] > 0:
                    x, y = smooth(columns[figure['x_axis']], columns[figure['y_axis']], figure['smooth'])
                    source = bokeh.models.ColumnDataSource({figure['x_axis']: x, figure['y_axis']: y})
                else:
                    source = bokeh.models.ColumnDataSource(bokeh.models.ColumnDataSource.from_df(columns))

                draw_function = group_figure.circle
                draw_args = {}

                if curve['style'] == 'line':

                    draw_function = group_figure.line
                    draw_args['line_color'] = curve['color']
                    draw_args['line_width'] = 2

                elif curve['style'] == 'square':

                    draw_function = group_figure.square
                    draw_args['color'] = curve['color']
                    draw_args['size'] = 10

                else:

                    draw_function = group_figure.circle
                    draw_args['color'] = curve['color']
                    draw_args['size'] = 10

                if show_legend:
                    draw_args['legend'] = curve['label']

                draw_function(
                        figure['x_axis'],
                        figure['y_axis'],
                        source=source,
                        **draw_args)

            group_figures.append(group_figure)

        group_grid = bokeh.layouts.gridplot([group_figures], title=get_title(group))
        bokeh.plotting.show(group_grid)

    print("Plotted in " + str(time.time() - start) + "s")

def render_table(data):
    from IPython.core.display import HTML, display
    display(HTML(data.to_html()))

if __name__ == '__main__':

    verbose = True
    read_all_results()
