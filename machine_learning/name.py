import os
import pathlib
from datetime import datetime

from machine_learning.misc import cd


def dataset_str(folder):
    """
    Get string representation for all current dataset in dataFiles
    This representation follows the "Letters and Numbers" rule explained in naming_schemes_v2.pptx in the naming branch
    :return: Dictionary of dataset as key and their representation as value
    """
    with cd(str(pathlib.Path(__file__).parent.parent.absolute()) + '/files/' + folder):  # Access folder with all dataset
    #with cd(data_path):  # Access folder with all dataset
        for roots, dirs, files in os.walk(os.getcwd()):
            data_dict = {}  # Dictionary of dataset and their first character
            for dataset in files:  # Loop through list of files
                if not dataset[0].isdigit():  # If first letter is not a number
                    data_dict[dataset] = dataset[0]  # Key as dataset as first character as value
                else:  # If first letter is a number
                    newstring = ''  # Empty string
                    for letter in dataset:  # Start looping through every character in the name
                        if letter.isdigit():  # If the character is a digit
                            newstring += letter  # Add number to the empty string
                        else:  # If letter is a string
                            newstring += letter  # Add string to the empty string
                            data_dict[dataset] = newstring
                            break  # Stop sequence at the first letter
    compare = []  # This list is for dataset that have unique first character
    repeat = []  # This list is for dataset that have matching first character
    duplicate_dict = {}  # Dictionary of dataset with the same first character
    for key in data_dict:  # For dataset in dictionary
        for value in data_dict[key]:  # For first character
            if value not in compare:  # If first character is not in our list
                compare.append(value)  # Add it to the empty list
            else:
                repeat.append(key)  # Add dataset that has matching first character to list
                duplicate_dict[value] = repeat  # Key as the string and list of dataset as values
    unique_list = []
    for key in duplicate_dict:  # For every key in duplicate dictionary
        count = 2  # Counter starts at 1
        unique = {}  # Final dictionary with all unique string representation for their respective dataset
        for duplicate in duplicate_dict[key]:  # For dataset with matching dataset_string
            data_dict.pop(duplicate, None)  # Remove values that have unique first character
            dataset_string = ''.join([duplicate[:-4][0], duplicate[:-4][-count:]])  # Combing first and last character
            if dataset_string not in unique.values():  # Check to see if the newly created string has duplicate
                unique[duplicate] = dataset_string   # Key as the dataset and the newly created string as value
            else:  # If the string still has duplicate
                count *= 2  # Increase counter by 1
                dataset_string = ''.join([duplicate[:-4][0], duplicate[:-4][-count:]])  # First, last and second to last
                unique[duplicate] = dataset_string  # Key as the dataset and the newly created string as value
                break  # Break the loop
        unique_list.append(unique)  # Get all dictionary for a situation that has multiple matching first character
    for dictionary in unique_list:  # Loop through all dictionaries
        data_dict.update(dictionary)  # Update the original dictionary
    print(data_dict)
    return data_dict


def algorithm_str():
    """
    Get string representation for all currently supported regressors
    This representation follows the "Letters and Numbers" rule explained in naming_schemes_v2.pptx in the naming branch
    :return:
    """
    algorithm_list = ['ada', 'rf', 'gdb', 'mlp', 'knn', 'nn', 'svm', 'cnn']  # Out current supported algorithm
    represent = [algor[0].upper() for algor in algorithm_list]  # List of algorithm's first letter with upper case
    algor_dict = {}
    for algor, rep in zip(algorithm_list, represent):  # Looping through two lists at the same time
        algor_dict[algor] = rep  # Key as algorithm and their string representation as value
    return algor_dict


def boolean_str(boolean):
    if boolean:
        return str(1)
    else:
        return str(0)


def name(algorithm, dataset, folder, featurized=False, tuned=False):
    """
    Give a unique name to a machine learning run
    :return: A unique name to a machine learning run
    
    Example: 
    
    Using rf, si_aerogel_AI_machine_readable_v2.csv, featurized, tuned on 06/17/2021 at 12:22:43 will give you: Rsv210_20210617-122243
    """

    algorithm_dict = algorithm_str()  # Get dictionary of algorithm and their string representation
    dataset_dict = dataset_str(folder)  # Get dictionary of dataset and their string representation
    if algorithm in algorithm_dict.keys():  # If the input algorithm is supported
        algorithm_string = algorithm_dict[algorithm]  # Get the string representation
    else:  # If it is not supported, give an error
        raise ValueError("The algorithm of your choice is not supported in our current workflow. Please use the "
                         "algorithms offered in grid.py ")
    if dataset in dataset_dict.keys():  # If input dataset is one that we have
        dataset_string = dataset_dict[dataset]  # Get the string representation
    else:  # If new dataset
        count = 2
        if dataset[:-4][0] in dataset_dict.values():  # If this dataset has the same first character as existing ones
            print("Duplicate First letter in input. Adding last and second to last characters")
            # Add first, last and second to last just to be safe
            dataset_string = ''.join([dataset[:-4][0], dataset[:-4][-count:]])
        else:  # If this dataset has a unique first character (compared to what we have in dataFiles)
            dataset_string = dataset[:-4][0]  # Dataset string will be its first character
    
    feat_string = boolean_str(featurized)
    tune_string = boolean_str(tuned)

    now = datetime.now()
    date_string = now.strftime("_%Y%m%d-%H%M%S")  # Get date and time string
    run_name = ''.join([algorithm_string, dataset_string, feat_string, tune_string, date_string])  # Run name
    date_str = now.strftime("%m/%d/%Y %H:%M:%S")
    print("Created {0} on {1}".format(run_name, date_str))
    return run_name
