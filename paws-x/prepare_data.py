import pandas as pd
import argparse
import os
import logging

logging.basicConfig(
    format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

parse = argparse.ArgumentParser()
parse.add_argument('--path', type=str, default='../../datasets/PAWS-X',
                    help='dataset folder path that contains train_0.tsv, dev_0.tsv and test_0.tsv')

args = parse.parse_args()

try:
    train_data = pd.read_csv(args.path+'/train_0.tsv', sep='\t', header=None, usecols=[0, 3], names=['Label', 'Text'])
    val_data = pd.read_csv(args.path+'/valid_0.tsv', sep='\t', header=None, usecols=[0, 3], names=['Label', 'Text'])
    test_data = pd.read_csv(args.path+'/test_0.tsv', sep='\t', header=None, usecols=[0, 3], names=['Label', 'Text'])
    train_data.to_json(args.path+'/train.json')
    val_data.to_json(args.path+'/dev.json')
    test_data.to_json(args.path+'/test.json')
except:
    print("enter a valid path that contains train_0.tsv, dev_0.tsv and test_0.tsv'")
    exit(0)