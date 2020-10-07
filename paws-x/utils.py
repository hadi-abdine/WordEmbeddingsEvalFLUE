from torch import nn
import torch.nn.functional as F
from torch import optim
import time
import numpy as np
import torch
from data_loader import tokenizerw2v, YelpDataset, collate_fn, get_loader
import os
import torch

np.random.seed(25)
torch.manual_seed(25)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
def masked_softmax(tensor, mask):
    tensor_shape = tensor.size()
    reshaped_tensor = tensor.view(-1, tensor_shape[-1])
    while mask.dim() < tensor.dim():
        mask = mask.unsqueeze(1)
    mask = mask.expand_as(tensor).contiguous().float()
    reshaped_mask = mask.view(-1, mask.size()[-1])
    result = nn.functional.softmax(reshaped_tensor * reshaped_mask, dim=-1)
    result = result * reshaped_mask
    result = result / (result.sum(dim=-1, keepdim=True) + 1e-10)
    return result.view(*tensor_shape)


def weighted_sum(tensor, weights, mask):
    weighted_sum = weights.bmm(tensor)

    while mask.dim() < weighted_sum.dim():
        mask = mask.unsqueeze(1)
    mask = mask.transpose(-1, -2)
    mask = mask.expand_as(weighted_sum).contiguous().float()

    return weighted_sum * mask

def get_mask(sequences_batch):
    sequence_len = 32
    batch_size = sequences_batch.size()[0]
    max_length = sequence_len
    mask = torch.ones(batch_size, max_length, dtype=torch.float)
    mask[sequences_batch[:, :max_length] == 2] = 0.0
    return mask

def replace_masked(tensor, mask, value):
    mask = mask.unsqueeze(1).transpose(2, 1)
    reverse_mask = 1.0 - mask
    values_to_add = value * reverse_mask
    return tensor * mask + values_to_add

   

def evaluate(model, data_loader, batch_size):
    count_batch = 0
    accuracy = 0
    for batch in data_loader:
        sequences1 = batch[0]
        sequences2 = batch[1]
        target = batch[2]
        out = model.forward(sequences1, sequences2)
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


def trainModel(model,  df_train, df_val, df_test, batch_size, modelw2v, sequence_len, printEvery=20, model_path=None, lr1=5e-5, training_epochs=30, checkpoints_path='../model', max_patience=4):
    epoch = 0
    loss = 0
    losses = []
    validation_accuracies = []
    count_iter = 0
    patience = 0 # before interrupting training 
    optimizer = optim.Adam(model.parameters(), lr=lr1)
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
    for i in range(epoch, training_epochs):
        print('-----EPOCH{}-----'.format(i+1))
        torch.manual_seed(25)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        loader = get_loader(df_train, batch_size, modelw2v, sequence_len)
        for batch in loader:
            torch.manual_seed(25)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
            loss += trainOneBatch(model, batch, optimizer, criterion)
            count_iter += 1
            if count_iter % printEvery == 0:
                time2 = time.time()
                print("Iteration: {0}, Time: {1:.4f} s, training loss: {2:.4f}".format(count_iter,
                                                                          time2 - time1, loss/printEvery))
                
                losses.append(loss)
                loss = 0 
        torch.manual_seed(25)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        validation_accuracy = evaluate(model, get_loader(df_val, batch_size, modelw2v, sequence_len), batch_size) 
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
            patience = 0
            torch.manual_seed(25)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
            t_acc = evaluate(model, get_loader(df_test, batch_size, modelw2v, sequence_len), batch_size)
            print('test accuracy: ', t_acc)
        else:
            patience += 1
            if patience == max_patience:
                print('Validation is not improving. stopping training')
                break
    print('final test accuracy: ', t_acc)             
    torch.save(model.state_dict(), os.path.join(checkpoints_path, 'model.pt')) 
    return losses, validation_accuracies 

def trainOneBatch(model, batch_input, optimizer, criterion):
    optimizer.zero_grad()
    sequences1 = batch_input[0] # get input sequence of shape: batch_size * sequence_len
    sequences2 = batch_input[1]
    targets = batch_input[2] # get targets of shape : batch_size
    torch.manual_seed(25)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    out = model.forward(sequences1, sequences2) # shape: batch_size * number_classes
    loss = criterion(out, targets)
    loss.backward() # compute the gradient
#     torch.nn.utils.clip_grad_norm_(model.parameters(), 0.1)
    optimizer.step() # update network parameters
    return loss.item() # return loss value/
    
class SoftmaxAttention(nn.Module):

    def forward(self, premise_batch, premise_mask, hypothesis_batch, hypothesis_mask):

        similarity_matrix = premise_batch.bmm(hypothesis_batch.transpose(2, 1).contiguous())

        prem_hyp_attn = masked_softmax(similarity_matrix, hypothesis_mask)
        hyp_prem_attn = masked_softmax(similarity_matrix.transpose(1, 2).contiguous(), premise_mask)

        attended_premises = weighted_sum(hypothesis_batch, prem_hyp_attn, premise_mask)
        attended_hypotheses = weighted_sum(premise_batch, hyp_prem_attn, hypothesis_mask)

        return attended_premises, attended_hypotheses