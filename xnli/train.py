import json
import gensim
from epoch_saver import EpochSaver
import torch
from nltk.tokenize import word_tokenize
from torch.utils.data import Dataset, DataLoader
from torch import nn
import torch.nn.functional as F
from torch import optim
import time
import numpy as np
import os
import logging
import argparse
from utils import trainModel, evaluate, loadModel, trainOneBatch
from fintuning_model import finetune
from gensim.models.wrappers import FastText

np.random.seed(1589387290)


logging.basicConfig(
    format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

parse = argparse.ArgumentParser()
parse.add_argument('--dataset', type=str, default='../../datasets/XNLI',
                    help='dataset folder path that contains validation, train and test set json files obtained from prepare_data')
parse.add_argument('--model_path', type=str, default='../../model/dascim1/dascim1.model',
                    help='directory to load the gensim embeddings model')
parse.add_argument('--algorithm', type=str, default='word2vec',
                    help='word2vec or fasttext')
parse.add_argument('--checkpoint', type=str, default='../../model',
                    help='oath to save the checkpoints')
parse.add_argument('--nb_epochs', type=int, default=30,
                    help='number of epochs')
parse.add_argument('--batch_size', type=int, default=32,
                    help='batch size')
parse.add_argument('--lr', type=float, default=5e-5,
                    help='learning rate')
parse.add_argument('--pretrained', type=str, default='yes')
parse.add_argument('--patience', type=int, default=4)


args = parse.parse_args()



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
sequence_len = 32
# training parameters 
batch_size = args.batch_size
training_epochs = args.nb_epochs
checkpoints_path = args.checkpoint
max_patience = args.patience # if the validation accuracy did not imporve for max_patience consecutive epochs: stop training 




try:
    with open(args.dataset+'/train.json', 'r') as f:
        df_train = json.load(f)
        f.close()

    with open(args.dataset+'/dev.json', 'r') as f:
        df_val = json.load(f)
        f.close()

    with open(args.dataset+'/test.json', 'r') as f:
        df_test = json.load(f)
        f.close()   
except:
    print("enter a valid path pr run prepare_data first!'")
    exit(0)    
    

    
# load the pretraind word vectors:
if args.algorithm == 'word2vec':
    try:
        modelw2v = gensim.models.Word2Vec.load(args.model_path, mmap='r')
        w2v = torch.FloatTensor(modelw2v.wv.vectors) 
    except:
        print("enter a valid model path or 'no' for non pretrained vectors...")
        exit(0)
elif args.algorithm == 'fasttext':
    try:
        modelw2v = FastText.load_fasttext_format(args.model_path)   
    except:
        print("enter a valid model path!")
        exit(0)
    w2v = torch.FloatTensor(modelw2v.wv.syn0)
else:
    print("enter word2vec or fasttext as algorithm")
    exit(0)


torch.cuda.empty_cache() 
torch.manual_seed(25)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
if args.pretrained == 'no':
    model1 = finetune(False, w2v)
else:
    model1 = finetune(True, w2v)
model1 = model1.to(device)
_, validation_accuracies = trainModel(model1, printEvery=240, model_path=None, lr1=args.lr, training_epochs=30, df_train=df_train, df_val=df_val, df_test=df_test, batch_size=batch_size, sequence_len=sequence_len, modelw2v=modelw2v, checkpoints_path=checkpoints_path, max_patience=max_patience)
    