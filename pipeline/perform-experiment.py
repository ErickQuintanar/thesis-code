import argparse
import json

# Environment variable where results are being stored
results_path = "/mnt/c/Users/erick/Dropbox/5.Semester/Thesis/Code/results"

def retrieve_config():
    parser = argparse.ArgumentParser(description='Input configuration file to perform experiment.')
    parser.add_argument('--file', dest='config', action='store_const', required=True, type=str,
                    help='Input a valid configuration file containing the details for the experiment to be performed.')
    
    args = parser.parse_args()

    try:
        f = open(args.config)
        # TODO: parse configuration
        config = parse_config(f)

        # TODO: Check if experiment has already beem performed
        check_previous_experiments(config)
    except FileNotFoundError:
        print('Configuration file not found.')
    
def parse_config(file):
    # TODO: extract json properties
    print("hello")

def check_previous_experiments(config):
    # TODO: parse configs from previous experiments and check that they are not the same, otherwise return values
    # from the previously performed experiment
    print("hello")

retrieve_config()
