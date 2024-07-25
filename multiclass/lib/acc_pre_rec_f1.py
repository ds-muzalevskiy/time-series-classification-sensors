# -*- coding: utf-8 -*-
"""
This file contains a function that can be called to evaluate accuracy, precision
recall and f1-score. Even though these functions are already included in
sklearn.metrics, we believe this to be necessary to ensure that all evaluations
are done in exactly the same fashion.
"""

import datetime
import os
import csv
import numpy
import time
from IPython.display import Javascript
from nbconvert import HTMLExporter
import codecs
import nbformat
from pathlib import Path


from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

ABSOLUTE_FILE_DIRECTORY = Path(os.path.abspath(__file__)).parent

# The following paths consider the structure of the project folder;
# this should not be changed unless project structure is changed
RELATIVE_PATH_TO_RESULT_FILES = "../06_results/02_result_files/"
RELATIVE_PATH_TO_NOTEBOOKS = "../06_results/01_jupyter_notebook_as_html/"

PATH_TO_RESULT_FILES = ABSOLUTE_FILE_DIRECTORY / RELATIVE_PATH_TO_RESULT_FILES
PATH_TO_NOTEBOOK_STORAGE = ABSOLUTE_FILE_DIRECTORY / RELATIVE_PATH_TO_NOTEBOOKS


def acc_pre_rec_f1(true_labels, predicted_labels):
    #
    # Input: true_labels: the true labels of the data which are part of
    #                     the original data
    #        predicted_labels: the labels predicted by the algorithm of choice;
    #                     be aware not to change the order of samples
    #
    # Comment on average parameter: This parameter is required for multiclass/
    #        multilabel targets. Calculate metrics for each label, and find their
    #        unweighted mean. This does not take label imbalance into account.
    #
    return (accuracy_score(true_labels, predicted_labels), 
            precision_score(true_labels, predicted_labels, average='macro'), 
            recall_score(true_labels, predicted_labels, average='macro'), 
            f1_score(true_labels, predicted_labels, average='macro'))


def save_notebook():
    return display(Javascript("IPython.notebook.save_notebook()"),
                   include=['application/javascript'])


def fail_if_not_valid(evaluation_tuple, dataset):
    # Input: evaluation_tuple: tuple of 4 float values; evaluation_tuple could be the acc_pre_rec_f1
    #           function which is defined above for proper input labels
    #        dataset: string variable that defines which data set the results
    #           belong to. this is a manual since I have no good idea of
    #           how to obtain this information from the notebook
    
    if (evaluation_tuple is None) & (dataset is None):
        raise ValueError('''Please input evaluation_tuple as a 4-tuple of scores and dataset as
              a string defining the dataset you are working on.''')
    elif (not evaluation_tuple is None) & (dataset is None):
        raise ValueError('''Please enter a string with the name of the data set ''')
    elif (evaluation_tuple is None) & (not dataset is None):
        raise ValueError('''Please input evaluation_tuple as a 4-tuple of scores''')
    else:
        pass
    
    if len(evaluation_tuple) != 4:
        raise ValueError('''Input does not match required format. It has to be of
              the form (acc, pre, rec, f1)!
              ''')
    for ele in evaluation_tuple:
        if (type(ele) != float) & (type(ele) != numpy.float64):
            raise ValueError('''Only float can be exported to result file''')

    
def export_acc_pre_rec_f1(x, dataset):
    #
    # This functions intent is to take the results of a single experiment
    # and exports them to another directory.
    #
    # Input: x: tuple of 4 float values; x could be the acc_pre_rec_f1
    #           function which is defined above for proper input labels
    #        dataset: string variable that defines which data set the results
    #           belong to. this is a manual since I have no good idea of
    #           how to obtain this information from the notebook
    # Filename is composed of date and time and current notebook name.
    # save function has to be run prior to this to get the correct
    # notebook in case there is more than one.
    # path for new .csv is given relative to working directory, this 
    # assumes that notebook is at correct directory in repo

    time_as_string = get_time_string()

    nb_name = get_newest_notebook_name_in_current_dir()
    filename = PATH_TO_RESULT_FILES / (time_as_string + '__' + nb_name + ".csv")

    x_with_meta_data = [time_as_string, nb_name, dataset]
    for ele in x:
        x_with_meta_data.append(ele)

    with open(filename, 'w') as csvfile:
        filewriter = csv.writer(csvfile)
        filewriter.writerow(x_with_meta_data)


def get_time_string():
    now = datetime.datetime.now()
    time_as_string = now.strftime("%Y-%m-%d_%H-%M-%S")
    return time_as_string


def save_latest_nb_as_html():
    #
    # This function copies the html version of the current notebook to
    # another directory. The name of the html is chosen such that it
    # can be linked to the result file generated by the
    # export_acc_pre_rec_f1 function
    #
    # Input: None: The file to copy is automatically found.
    now_as_string = get_time_string()
    nb_name = get_newest_notebook_name_in_current_dir()

    output_HTML(nb_name + '.ipynb',
             PATH_TO_NOTEBOOK_STORAGE / (now_as_string + '__' + nb_name + ".html"))


def output_HTML(read_file, output_file):

    exporter = HTMLExporter()
    # read_file is '.ipynb', output_file is '.html'
    output_notebook = nbformat.read(read_file, as_version=4)
    output, resources = exporter.from_notebook_node(output_notebook)
    codecs.open(output_file, 'w', encoding='utf-8').write(output)


def get_newest_notebook_name_in_current_dir():
    notebooks = [ele for ele in os.listdir() if ele.endswith('.ipynb')]
    notebooks.sort(key=lambda x: os.path.getmtime(x))
    nb_name = notebooks[-1].split('.')[0]
    return nb_name

    
def export_results_and_copy_notebook(evaluation_results, dataset):
    #
    # This function combines the functions to export the results
    # and the function to copy the html of the notebook.
    #
    # Input: evaluation_results: tuple of 4 float values, forwarded to export_acc_pre_rec_f1
    
    save_notebook()
    time.sleep(3)
    
    fail_if_not_valid(evaluation_results, dataset)
    export_acc_pre_rec_f1(evaluation_results, dataset)
    save_latest_nb_as_html()

    
    
    
    
    
    
