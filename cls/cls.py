import pandas as pd
import argparse
import os
import logging
import json
import gensim
from gensim.models.wrappers import FastText
from epoch_saver import EpochSaver
import torch
from nltk.tokenize import word_tokenize
from torch.utils.data import Dataset, DataLoader
from torch import nn
import torch.nn.functional as F
from torch import optim
import time
import numpy as np

logging.basicConfig(
    format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

parse = argparse.ArgumentParser()
parse.add_argument('--dataset', type=str, default='../../datasets/CLS/books',
                    help='dataset folder path that contains validation, traind and test set tsv files')
parse.add_argument('--model_path', type=str, default='../../model/dascim1/dascim1.model',
                    help='directory to load the gensim embeddings model')
parse.add_argument('--algorithm', type=str, default='word2vec',
                    help='word2vec or fasttext')
parse.add_argument('--checkpoint', type=str, default='../../model',
                    help='path to save the checkpoints')
parse.add_argument('--nb_epochs', type=int, default=40,
                    help='number of epochs')
parse.add_argument('--batch_size', type=int, default=64,
                    help='batch size')
parse.add_argument('--patience', type=int, default=4)
parse.add_argument('--lr', type=float, default=5e-6,
                    help='learning rate')
parse.add_argument('--pretrained', type=str, default='yes')

args = parse.parse_args()
np.random.seed(1589387290)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
sequence_len = 512
# training parameters 
batch_size = args.batch_size
training_epochs = args.nb_epochs
checkpoints_path = args.checkpoint
max_patience = args.patience # if the validation accuracy did not imporve for max_patience consecutive epochs: stop training 

# prepare the dataset to use it in our model:
try:
    train_data = pd.read_csv(args.dataset+'/train.tsv', sep='\t')
    val_data = pd.read_csv(args.dataset+'/dev.tsv', sep='\t')
    test_data = pd.read_csv(args.dataset+'/test.tsv', sep='\t')
    train_data.to_json(args.dataset+'/train.json')
    val_data.to_json(args.dataset+'/dev.json')
    test_data.to_json(args.dataset+'/test.json')
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
    print("enter a valid folder path!")
    exit(0)


# load the pretraind word vectors:
if args.algorithm == 'word2vec':
    try:
        modelw2v = gensim.models.Word2Vec.load(args.model_path, mmap='r')
        w2v = torch.FloatTensor(modelw2v.wv.vectors) 
    except:
        print("enter a valid model path!")
        exit(0)
elif args.algorithm == 'fasttext':
    try:
        modelw2v = FastText.load_fasttext_format(args.model_path)
        w2v = torch.FloatTensor(modelw2v.wv.syn0)
    except:
        print("enter a valid model path!")
        exit(0)
else:
    print("enter word2vec or fasttext as algorithm")
    exit(0)



# define a tokenizer function :
def tokenizerw2v(sentence):
    sep = word_tokenize(sentence)
    tokenized = []
    for word in sep:
        try:
            tokenized.append(modelw2v.wv.vocab[word].index)
        except:
            assert True
    if len(tokenized) > sequence_len:
        tokenized = tokenized[:sequence_len-1]
        tokenized.append(2)
    else:
        for i in range(sequence_len-len(tokenized)):
            tokenized.append(2)
    return tokenized



#define the data loader:
class FrDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data['Text'])

    def __getitem__(self, index):
        content = self.data['Text'][str(index)]
        content = tokenizerw2v(content)
        category = self.data['Label'][str(index)]
        sample = {'Text': content, 'Label': category}
        return sample

def collate_fn(data):
    content_sequences = torch.zeros(size=(len(data), sequence_len),
                                   dtype=torch.long, device=device)
    for i in range(len(data)):
        sequence = data[i]['Text']
        content_sequences[i] = torch.tensor(sequence)
    categories = torch.tensor([el['Label'] for el in data],
                              dtype=torch.long, device=device)
    return content_sequences, categories

def get_loader(data, batch_size=5):
    """
    Args:
        path: path to dataset.
        batch_size: mini-batch size.
    Returns:
        data_loader: data loader for custom dataset.
    """
    dataset = FrDataset(data)
    data_loader = torch.utils.data.DataLoader(dataset=dataset,
                                              batch_size=batch_size,
                                              shuffle=False,
                                              collate_fn=collate_fn,
                                              drop_last=True,
                                              )

    return data_loader


#define the finetuning model:
class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        if args.pretrained != 'no':
            self.embedding = nn.Embedding.from_pretrained(w2v)
        else:
            self.embedding = nn.Embedding(w2v.shape[0], w2v.shape[1])
        self.lstm = nn.LSTM(input_size=self.embedding.embedding_dim,
                            hidden_size=1500,
                            num_layers=2, 
                            bidirectional=True,
                            batch_first=True)
        self.dropout1 = nn.Dropout(p=0.1)
        self.linear1 = nn.Linear(3000, 3000)
        self.tanh = nn.Tanh()
        self.dropout2 = nn.Dropout(p=0.1)
        self.linear2 = nn.Linear(3000, 2)
        self.softmax = nn.LogSoftmax(-1)
    
    def forward(self, input_tensor):
        x = self.embedding(input_tensor)
        lstm_out, (ht, ct) = self.lstm(x)
        x = self.dropout1(torch.cat((ht[-2,:,:], ht[-1,:,:]), dim = 1))
        x = self.linear1(x)
        x = self.tanh(x)
        x = self.dropout2(x)
        out = self.linear2(x)
        return self.softmax(out)


#define the evaluation metric (accuracy):
def evaluate(model, data_loader):
    count_batch = 0
    accuracy = 0
    for batch in data_loader:
        sequences = batch[0]
        target = batch[1]
        out = model.forward(sequences)
        out = out.detach()
        predicted = torch.argmax(out, -1)
        accuracy += torch.sum(predicted==target).item() / batch_size
        count_batch += 1
    accuracy = accuracy/count_batch
    return accuracy


def loadModel(model, optimizer, model_path):
    checkpoint = torch.load(model_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch'] + 1
    best_validation_accuracy  = checkpoint['validation_accuracy']
    loss = checkpoint['loss']
    return epoch, best_validation_accuracy, loss
   

def trainModel(model, printEvery=20, model_path=None):
    epoch = 0
    loss = 0
    losses = []
    validation_accuracies = []
    count_iter = 0
    patience = 0 # before interrupting training 
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    #negative log likelihood
    criterion = nn.NLLLoss()
    best_validation_accuracy = 0 # to decide whether to checkpoint or not
    best_model_checkpoint = ''
    #load model from checkpoint
    if model_path != None:
        epoch, best_validation_accuracy, loss = loadModel(model,optimizer, model_path)

    time1 = time.time()
    training_accuracy_epochs = [] # save training accuracy for each epoch
    validation_accuracy_epochs = [] # save validation accuracy for each epoch 
    t_acc = 0.0
    for i in range(epoch, training_epochs):
        print('-----EPOCH{}-----'.format(i+1))
        loader = get_loader(df_train, batch_size)
        for batch in loader:
            loss += trainOneBatch(model, batch, optimizer, criterion)
            count_iter += 1
            if count_iter % printEvery == 0:
                time2 = time.time()
                print("Iteration: {0}, Time: {1:.4f} s, training loss: {2:.4f}".format(count_iter,
                                                                          time2 - time1, loss/printEvery))
                
                losses.append(loss)
                loss = 0 
        validation_accuracy = evaluate(model, get_loader(df_val, batch_size)) 
        validation_accuracy_epochs.append(validation_accuracy)
        best_validation_accuracy = max(best_validation_accuracy, validation_accuracy)
        print('Epoch {0} done: validation_accuracy = {1:.3f}'.format(i+1, validation_accuracy))

        validation_accuracies.append(validation_accuracy)
        if validation_accuracy == best_validation_accuracy:
            try:
                os.remove(os.path.join(checkpoints_path, best_model_checkpoint))
            except:
                print('not found')
            best_model_checkpoint = 'checkpoint_epoch_{}'.format(i)
            print('validation accuracy improved: saving checkpoint...')
            save_path = os.path.join(checkpoints_path, best_model_checkpoint)
            torch.save({
            'epoch': i,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': loss,
            'validation_accuracy': validation_accuracy
            }, save_path)
            print('checkpoint saved to: {}'.format(save_path))
            torch.manual_seed(25)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
            t_acc = evaluate(model, get_loader(df_test, batch_size))
            print('test_acc on best model: ',t_acc)
            patience = 0
        else:
            patience += 1
            if patience == max_patience:
                print('Validation is not improving. stopping training')
                break
    print('test accuracy = ', t_acc)
#     torch.save(model.state_dict(), os.path.join(checkpoints_path, 'model.pt')) 
    return losses, validation_accuracies 

def trainOneBatch(model, batch_input, optimizer, criterion):
    torch.manual_seed(25)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    optimizer.zero_grad()
    sequences = batch_input[0] # get input sequence of shape: batch_size * sequence_len
    targets = batch_input[1] # get targets of shape : batch_size
    out = model.forward(sequences) # shape: batch_size * number_classes
    loss = criterion(out, targets)
    loss.backward() # compute the gradient
    optimizer.step() # update network parameters
    return loss.item() # return loss value/


torch.manual_seed(25)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
model = Model()
model = model.to(device)
_, validation_accuracies = trainModel(model, 20, model_path=None)

print("\n Finished !!!!! see you soon...")