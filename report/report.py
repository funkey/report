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

processed_dir = 'processed'
verbose = False

def fetch_data_frame_from_json():

    start = time.time()

    print("Collecting record files...")
    record_files = []
    for root, _, filenames in os.walk(processed_dir):
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

def read_all_results(data_backend = 'json_files'):
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
        data_frame = fetch_data_frame_from_json()
    else:
        raise RuntimeError("no such data backend '" + str(data_backend) + "'")

    vs = data_frame['voi_split']
    vm = data_frame['voi_merge']
    rs = data_frame['rand_split']
    rm = data_frame['rand_merge']

    data_frame['voi_sum'] = np.array(vs + vm, dtype=np.float32)
    data_frame['arand'] = np.array(1.0 - (2.0*rs*rm)/(rs+rm), dtype=np.float32)
    data_frame['cremi_score'] = np.sqrt(np.array(data_frame['voi_sum']*data_frame['arand'], dtype=np.float32))

    return data_frame

def matches(record, configuration):

    # symbolic group 'average' matches all
    if isinstance(configuration, str) and configuration == 'average':
        return True

    for key in configuration.keys():

        # skip non-record fields
        if key in ['label', 'color']:
            continue

        if key not in record:
            return False

        allowed_values = configuration[key]
        if not isinstance(allowed_values, list):
            allowed_values = [allowed_values]

        if record[key] not in allowed_values:
            return False

    return True

def get_configuration_label(configuration):

    if 'label' in configuration:
        return configuration['label']

    label = ""
    prefix = ""
    for key in configuration.keys():

        if key == 'color':
            continue

        if not isinstance(configuration[key], list):
            label += prefix + str(configuration[key])
            prefix = " "

    return label

def get_title(group):

    if 'title' in group:
        return str(group['title'])
    return str(group)

def filter(records, configurations):

    match_mask = None

    values_queries = []
    for c in configurations:
        for k in c:

            if k in ['color', 'label']:
                continue

            possible_values = c[k] if isinstance(c[k], list) else [c[k]]

            # first possible value
            value_mask = (records[k] == possible_values[0])
            for i in range(2, len(possible_values)):
                value_mask |= (records[k] == possible_values[i])

            values_queries.append(" or ".join([str(k) + " == " + str(v) for v in possible_values]))

            if match_mask is None:
                match_mask = value_mask
            else:
                match_mask &= value_mask

    query_string = " and ".join(values_queries)

    if verbose:
        print("Filtering with: " + query_string)

    filtered = records[match_mask]
    return filtered

def plot(groups, figures, configurations, all_records):

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

    # data preparation
    start = time.time()

    print("Preparing plot data")
    start = time.time()

    plots = { get_title(group): [] for group in groups }

    for configuration in configurations:

        # attach to the plots
        for group in groups:

            filtered_records = filter(all_records, [configuration, group])

            if verbose:
                print("filtered records for " + str(configuration) + ", group " + str(group) + ":")
                print(filtered_records)

            # create plot with meta-data
            plot = {

                'columns': filtered_records,
                'label': get_configuration_label(configuration),
                'color': bokeh.palettes.Spectral6[hash(str(configuration))%len(bokeh.palettes.Spectral6)] if 'color' not in configuration else configuration['color']
            }

            plots[get_title(group)].append(plot)

    print("Prepared data in " + str(time.time() - start) + "s")

    # plotting
    start = time.time()

    average_plot = {}
    for group in groups:

        group_figures = []
        for figure in figures:

            keys = list(filtered_records.keys())
            keys.sort()
            tooltips="".join([
                "<div><span>%s: @%s</span></div>"%(key,key) for key in keys
            ])
            hover = bokeh.models.HoverTool(tooltips=tooltips)
            tools = ['save','pan','wheel_zoom','box_zoom','reset']
            group_figure = bokeh.plotting.figure(
                    title=get_title(group) + " " + get_title(figure),
                    tools=[hover] + tools,
                    active_scroll='wheel_zoom',
                    x_axis_label=figure['x_axis'],
                    y_axis_label=figure['y_axis'])

            if isinstance(group, str) and group == 'average':
                group_figure_plots = [ v for (_,v) in average_plot.iteritems() ]
            else:
                group_figure_plots = plots[get_title(group)]

            for plot in group_figure_plots:

                source = bokeh.models.ColumnDataSource(bokeh.models.ColumnDataSource.from_df(plot['columns']))

                group_figure.circle(
                        figure['x_axis'],
                        figure['y_axis'],
                        source=source,
                        legend=plot['label'],
                        line_color=plot['color'],
                        fill_color=plot['color'],
                        line_width=2,
                        size=10)

            group_figures.append(group_figure)

        group_grid = bokeh.layouts.gridplot([group_figures], title=get_title(group))
        bokeh.plotting.show(group_grid)

    print("Plotted in " + str(time.time() - start) + "s")

if __name__ == '__main__':

    verbose = True
    read_all_results()
