import os
import argparse

import json
from jsondiff import diff

from training.train_model import train_script

# TODO: Include results path in argument(?)

def perform_experiment(config, results_path="/mnt/c/Users/erick/Dropbox/5.Semester/Thesis/Code/results"):
    # Check if experiment has already been performed
    directory = results_path+"/configs/"+config["dataset"]+"/"
    for filename in os.listdir(directory):            
        # Open and load the JSON file
        with open(directory + filename, 'r') as file:
            # Compare data with config
            if filename == ".gitkeep":
                continue
            data = json.load(file)
            differences = diff(config, data)
            if len(differences.keys()) < 3:
                print("Experiment has already been performed with id = "+str(differences["id"]))
                print("Config is at: "+directory+filename)
                print("Weights are at: "+results_path+"/weights/"+config["dataset"]+"/"+filename[:-4]+"npy")
                return
    print("Performing experiment")
    train_script(config)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Input configuration file to perform experiment.')
    parser.add_argument('--config', dest='config', required=True, type=str,
                    help='Input a valid configuration file containing the details for the experiment to be performed.')
        
    args = parser.parse_args()
    
    # Parse configuration
    with open(args.config, 'r') as file:
        config = json.load(file)
        perform_experiment(config)