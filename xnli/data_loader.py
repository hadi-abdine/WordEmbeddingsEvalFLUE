from torch.utils.data import Dataset, DataLoader
import torch
from nltk.tokenize import word_tokenize
import numpy as np
np.random.seed(1589387290)

def tokenizerw2v(sentence, modelw2v, sequence_len):
    sentences = sentence.split('\t')
    sent1 =sentences[0]
    sent2 = sentences[1]
    sep1 = word_tokenize(sent1)
    sep2 = word_tokenize(sent2)
    tokenized1 = []
    tokenized2 = []
    for word in sep1:
        try:
            tokenized1.append(modelw2v.wv.vocab[word].index)
        except:
            assert True
    if len(tokenized1) > sequence_len:
        tokenized1 = tokenized1[:sequence_len-1]
        tokenized1.append(2)
    else:
        for i in range(sequence_len-len(tokenized1)):
            tokenized1.append(2)
    for word in sep2:
        try:
            tokenized2.append(modelw2v.wv.vocab[word].index)
        except:
            assert True
    if len(tokenized2) > sequence_len:
        tokenized2 = tokenized1[:sequence_len-1]
        tokenized2.append(2)
    else:
        for i in range(sequence_len-len(tokenized2)):
            tokenized2.append(2)
    if tokenized2[0]==2:
        tokenized2[0] = 0
    if tokenized1[0]==2:
        tokenized1[0] = 0
    return tokenized1, tokenized2   

class YelpDataset(Dataset):
    torch.manual_seed(25)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    def __init__(self, data, modelw2v, sequence_len):
        self.data = data
        self.sequence_len = sequence_len
        self.modelw2v = modelw2v

    def __len__(self):
        return len(self.data['Text'])

    def __getitem__(self, index):
        content = self.data['Text'][str(index)]
        content1, content2 = tokenizerw2v(content, self.modelw2v, self.sequence_len)
        category = self.data['Label'][str(index)]
        sample = {'Text1': content1, 'Text2':content2, 'Label': category}
        return sample

def collate_fn(data):
    torch.manual_seed(25)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    sequence_len = 32
    content_sequences1 = torch.zeros(size=(len(data), sequence_len),
                                   dtype=torch.long, device=device)
    content_sequences2 = torch.zeros(size=(len(data), sequence_len),
                                   dtype=torch.long, device=device)
    for i in range(len(data)):
        sequence1 = data[i]['Text1']
        sequence2 = data[i]['Text2']
        content_sequences1[i] = torch.tensor(sequence1)
        content_sequences2[i] = torch.tensor(sequence2)
    categories = torch.tensor([el['Label'] for el in data],
                              dtype=torch.long, device=device)
    return content_sequences1, content_sequences2, categories

def get_loader(data, batch_size, modelw2v, sequence_len):
    """
    Args:
        path: path to dataset.
        batch_size: mini-batch size.
    Returns:
        data_loader: data loader for custom dataset.
    """
    torch.manual_seed(25)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    dataset = YelpDataset(data, modelw2v, sequence_len)
    sequence_len1=sequence_len
    data_loader = torch.utils.data.DataLoader(dataset=dataset,
                                              batch_size=batch_size,
                                              shuffle=False,
                                              collate_fn=collate_fn,
                                              drop_last=True,
                                              )

    return data_loader