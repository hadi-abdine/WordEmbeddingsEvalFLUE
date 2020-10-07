from torch import nn
import torch
import torch.nn.functional as F
from utils import SoftmaxAttention, masked_softmax, weighted_sum, get_mask, replace_masked
import numpy as np



class Seq2SeqEncoder(nn.Module):
    def __init__(self,
                 rnn_type,
                 input_size,
                 hidden_size,
                 num_layers=1,
                 bias=True,
                 dropout=0.0,
                 bidirectional=False):

        assert issubclass(rnn_type, nn.RNNBase),\
            "rnn_type must be a class inheriting from torch.nn.RNNBase"

        super(Seq2SeqEncoder, self).__init__()

        self.rnn_type = rnn_type
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bias = bias
        self.dropout = dropout
        self.bidirectional = bidirectional

        self._encoder = rnn_type(input_size,
                                 hidden_size,
                                 num_layers=num_layers,
                                 bias=bias,
                                 batch_first=True,
                                 dropout=dropout,
                                 bidirectional=bidirectional)

    def forward(self, sequences_batch):
        outputs, _ = self._encoder(sequences_batch, None)
        return outputs
    
    
class finetune(nn.Module):
    
    def __init__(self, pretrained, w2v):
        np.random.seed(1589387290)
        torch.manual_seed(25)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        super(finetune, self).__init__()
        if pretrained:
            self.embedding = nn.Embedding.from_pretrained(w2v)
        else:
            self.embedding = nn.Embedding(w2v.shape[0], w2v.shape[1])
        
        
        self._encoding = Seq2SeqEncoder(nn.LSTM,
                                        self.embedding.embedding_dim,
                                        1500,
                                        bidirectional=True)
        
        self._attention = SoftmaxAttention()
        self._projection = nn.Sequential(nn.Linear(4*2*1500,
                                                   1500),
                                         nn.ReLU())
        
        self.dropout = nn.Dropout(p=0.1)

        self._composition = Seq2SeqEncoder(nn.LSTM,
                                           1500,
                                           1500,
                                           bidirectional=True)


        self._classification = nn.Sequential(nn.Dropout(p=0.1),
                                             nn.Linear(2*4*1500,
                                                       1500),
                                             nn.Tanh(),
                                             nn.Dropout(p=0.1),
                                             nn.Linear(1500,
                                                       3))
        
    
    def forward(self, input_tensor1, input_tensor2):
        np.random.seed(1589387290)
        torch.manual_seed(25)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        x1_mask = get_mask(input_tensor1).to(self.device)
        x2_mask = get_mask(input_tensor2).to(self.device)
        
        x1 = self.embedding(input_tensor1)
        x2 = self.embedding(input_tensor2)
        
        x1_encoded = self._encoding(x1)
        x2_encoded = self._encoding(x2)

        attended_x1, attended_x2 = self._attention(x1_encoded, x1_mask, x2_encoded, x2_mask)
        
        enhanced_x1 = torch.cat([x1_encoded,
                                 attended_x1,
                                 x1_encoded - attended_x1,
                                 x1_encoded * attended_x1],
                                 dim=-1)
        enhanced_x2 = torch.cat([x2_encoded,
                                 attended_x2,
                                 x2_encoded - attended_x2,
                                 x2_encoded * attended_x2],
                                 dim=-1)
        
        
        projected_x1 = self._projection(enhanced_x1)
        projected_x2 = self._projection(enhanced_x2)
        
        v_ai = self._composition(projected_x1)
        v_bj = self._composition(projected_x2)
        
        v_a_avg = torch.sum(v_ai * x1_mask.unsqueeze(1).transpose(2, 1), dim=1) / torch.sum(x1_mask, dim=1, keepdim=True)
        v_b_avg = torch.sum(v_bj * x2_mask.unsqueeze(1).transpose(2, 1), dim=1) / torch.sum(x2_mask, dim=1, keepdim=True)

        v_a_max, _ = replace_masked(v_ai, x1_mask, -1e7).max(dim=1)
        v_b_max, _ = replace_masked(v_bj, x2_mask, -1e7).max(dim=1)

        v = torch.cat([v_a_avg, v_a_max, v_b_avg, v_b_max], dim=1)

        logits = self._classification(v)
        probabilities = nn.functional.log_softmax(logits, dim=-1)

        return probabilities