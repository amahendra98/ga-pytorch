import os
import numpy as np
import matplotlib
import flag_reader

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns;

sns.set()
from sklearn.neighbors import NearestNeighbors
from pandas.plotting import table
from scipy.spatial import distance_matrix
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import minimum_spanning_tree

class HMpoint(object):
    """
    This is a HeatMap point class where each object is a point in the heat map
    properties:
    1. BV_loss: best_validation_loss of this run
    2. feature_1: feature_1 value
    3. feature_2: feature_2 value, none is there is no feature 2
    """

    def __init__(self, bv_loss, f1, f2=None, f1_name='f1', f2_name='f2'):
        self.bv_loss = bv_loss
        self.feature_1 = f1
        self.feature_2 = f2
        self.f1_name = f1_name
        self.f2_name = f2_name
        # print(type(f1))

    def to_dict(self):
        return {
            self.f1_name: self.feature_1,
            self.f2_name: self.feature_2,
            self.bv_loss: self.bv_loss
        }


def HeatMapBVL(plot_x_name, plot_y_name, title, save_name='HeatMap.png', HeatMap_dir='HeatMap',
               feature_1_name=None, feature_2_name=None,
               heat_value_name='best_validation_loss'):
    """
    Plotting a HeatMap of the Best Validation Loss for a batch of hyperswiping thing
    First, copy those models to a folder called "HeatMap"
    Algorithm: Loop through the directory using os.look and find the parameters.txt files that stores the
    :param HeatMap_dir: The directory where the checkpoint folders containing the parameters.txt files are located
    :param feature_1_name: The name of the first feature that you would like to plot on the feature map
    :param feature_2_name: If you only want to draw the heatmap using 1 single dimension, just leave it as None
    """
    one_dimension_flag = False  # indication flag of whether it is a 1d or 2d plot to plot
    # Check the data integrity
    if (feature_1_name == None):
        print("Please specify the feature that you want to plot the heatmap");
        return
    if (feature_2_name == None):
        one_dimension_flag = True
        print("You are plotting feature map with only one feature, plotting loss curve instead")

    # Get all the parameters.txt running related data and make HMpoint objects
    HMpoint_list = []
    df_list = []  # make a list of data frame for further use
    for subdir, dirs, files in os.walk(HeatMap_dir):
        for file_name in files:
            if (file_name == 'parameters.txt'):
                file_path = os.path.join(subdir, file_name)  # Get the file relative path from
                # df = pd.read_csv(file_path, index_col=0)
                with open(file_path, 'r') as f:
                    flag_dict = eval(f.read())
                #flag = flag_reader.load_flags(subdir)
                #flag_dict = vars(flag)
                df = pd.DataFrame()
                for k in flag_dict:
                    print(k)
                    df[k] = pd.Series(str(flag_dict[k]), index=[0])
                print(df)
                if (one_dimension_flag):
                    # print(df[[heat_value_name, feature_1_name]])
                    # print(df[heat_value_name][0])
                    # print(df[heat_value_name].iloc[0])
                    df_list.append(df[[heat_value_name, feature_1_name]])
                    HMpoint_list.append(HMpoint(float(df[heat_value_name][0]), eval(str(df[feature_1_name][0])),
                                                f1_name=feature_1_name))
                else:
                    if feature_2_name == 'linear_unit':  # If comparing different linear units
                        df['linear_unit'] = eval(df[feature_1_name][0])[1]
                        df['best_validation_loss'] = get_bvl(file_path)
                    if feature_2_name == 'kernel_second':  # If comparing different kernel convs
                        print(df['conv_kernel_size'])
                        print(type(df['conv_kernel_size']))
                        df['kernel_second'] = eval(df['conv_kernel_size'][0])[1]
                        df['kernel_first'] = eval(df['conv_kernel_size'][0])[0]
                    df_list.append(df[[heat_value_name, feature_1_name, feature_2_name]])
                    HMpoint_list.append(HMpoint(float(df[heat_value_name][0]), eval(str(df[feature_1_name][0])),
                                                eval(str(df[feature_2_name][0])), feature_1_name, feature_2_name))

    print(df_list)
    # Concatenate all the dfs into a single aggregate one for 2 dimensional usee
    df_aggregate = pd.concat(df_list, ignore_index=True, sort=False)
    # print(df_aggregate[heat_value_name])
    # print(type(df_aggregate[heat_value_name]))
    df_aggregate.astype({heat_value_name: 'float'})
    # print(type(df_aggregate[heat_value_name]))
    # df_aggregate = df_aggregate.reset_index()
    print("before transformation:", df_aggregate)
    [h, w] = df_aggregate.shape
    for i in range(h):
        for j in range(w):
            if isinstance(df_aggregate.iloc[i, j], str) and (isinstance(eval(df_aggregate.iloc[i, j]), list)):
                # print("This is a list!")
                df_aggregate.iloc[i, j] = len(eval(df_aggregate.iloc[i, j]))

    print("after transoformation:", df_aggregate)

    # Change the feature if it is a tuple, change to length of it
    for cnt, point in enumerate(HMpoint_list):
        print("For point {} , it has {} loss, {} for feature 1 and {} for feature 2".format(cnt,
                                                                                            point.bv_loss,
                                                                                            point.feature_1,
                                                                                            point.feature_2))
        assert (isinstance(point.bv_loss, float))  # make sure this is a floating number
        if (isinstance(point.feature_1, tuple)):
            point.feature_1 = len(point.feature_1)
        if (isinstance(point.feature_2, tuple)):
            point.feature_2 = len(point.feature_2)

    f = plt.figure()
    # After we get the full list of HMpoint object, we can start drawing
    if (feature_2_name == None):
        print("plotting 1 dimension HeatMap (which is actually a line)")
        HMpoint_list_sorted = sorted(HMpoint_list, key=lambda x: x.feature_1)
        # Get the 2 lists of plot
        bv_loss_list = []
        feature_1_list = []
        for point in HMpoint_list_sorted:
            bv_loss_list.append(point.bv_loss)
            feature_1_list.append(point.feature_1)
        print("bv_loss_list:", bv_loss_list)
        print("feature_1_list:", feature_1_list)
        # start plotting
        plt.plot(feature_1_list, bv_loss_list, 'o-')
    else:  # Or this is a 2 dimension HeatMap
        print("plotting 2 dimension HeatMap")
        # point_df = pd.DataFrame.from_records([point.to_dict() for point in HMpoint_list])
        df_aggregate = df_aggregate.reset_index()
        df_aggregate.sort_values(feature_1_name, axis=0, inplace=True)
        df_aggregate.sort_values(feature_2_name, axis=0, inplace=True)
        df_aggregate.sort_values(heat_value_name, axis=0, inplace=True)
        print("before dropping", df_aggregate)
        df_aggregate = df_aggregate.drop_duplicates(subset=[feature_1_name, feature_2_name], keep='first')
        print("after dropping", df_aggregate)
        point_df_pivot = df_aggregate.reset_index().pivot(index=feature_1_name, columns=feature_2_name,
                                                          values=heat_value_name).astype(float)
        point_df_pivot = point_df_pivot.rename({'5': '05'}, axis=1)
        point_df_pivot = point_df_pivot.reindex(sorted(point_df_pivot.columns), axis=1)
        print("pivot=")
        csvname = HeatMap_dir + 'pivoted.csv'
        point_df_pivot.to_csv(csvname)
        print(point_df_pivot)
        sns.heatmap(point_df_pivot, cmap="YlGnBu")
    plt.xlabel(plot_y_name)  # Note that the pivot gives reversing labels
    plt.ylabel(plot_x_name)  # Note that the pivot gives reversing labels
    plt.title(title)
    plt.savefig(save_name)


if __name__ == '__main__':
    directory = 'results/sweeps/Heat_Maps/'

    '''
    identifier = 'Mutation_0.01_Truncation-Ratio_0.3'
    HeatMapBVL("Population", "K-Nearest Neighbor", 'Population vs. K-Nearest Neighbor',
                       HeatMap_dir=directory, save_name=identifier+'.png', feature_1_name='pop_size',
                       feature_2_name='k', heat_value_name='suc_gen')
    '''

    folders = os.listdir(directory)
    for f in folders:
        if f.find('Truncation') != -1:
            pass
            HeatMapBVL("Population", "K-Nearest Neighbor", 'Population vs. K-Nearest Neighbor',
                       HeatMap_dir=directory+'/'+f, save_name=f+'.png', feature_1_name='pop_size',
                       feature_2_name='k', heat_value_name='suc_gen')
        if f.find('K-Ratio') != -1:
            HeatMapBVL("Population", "Truncation Ratio", 'Population vs. Truncation Ratio',
                       HeatMap_dir=directory+'/'+f, save_name=f+'.png', feature_1_name='pop_size',
                       feature_2_name='trunc_threshold', heat_value_name='suc_gen')
        if f.find('Population') != -1:
            pass
            HeatMapBVL("Truncation Ratio", "K-Nearest Neighbor Ratio", 'Truncation Ratio vs. K-Nearest Neighbor',
                       HeatMap_dir=directory+'/'+f, save_name=f+'.png', feature_1_name='k',
                       feature_2_name='trunc_threshold', heat_value_name='suc_gen')