# Evaluation of Word Embeddings on FLUE Benchmark

This repository includes the codes to evaluate the quality of French word embeddings on three NLU tasks from FLUE: CLS (Sentiment classification), PAWSX (Paraphrasing) and XNLI (Inference). This repository can be used to reproduce the results obtained in [https://arxiv.org/abs/2105.01990] 

To prepare the data for the three datasets we refer to the original repository of FlauBERT and FLUE: [https://github.com/getalp/Flaubert/tree/master/flue] <br>

Our French word embeddings can be downloaded from [http://master2-bigdata.polytechnique.fr/FrenchLinguisticResources/resources] upon request.

To evaluate the quality of the static word embeddings on the sentiment classification task we will use feed the embeddings into a BiLSTM model while for PAWSX and XNLI tasks we will use ESIM. All the details about the models can be found in the above mentioned paper.

### Instructions 
To finetune the static word embeddings on these tasks, you must provide the path to the model (binary file), the path to the data folder containing 3 json files: train.json, dev.json and test.json, specify whether the word embeddings are trained with word2vec or fasttext, give a checkpoints save path, number of epochs, patience, batch size and learning rate. 

For example:
```ruby
python3 cls.py \
        --dataset /path/to/dataset/folder/  \
        --model_path /path/to/model.bin  \
        --algorithm word2vec  \
        --checkpoint /path/to/checkpoint_folder \
        --nb_epochs 30  \
        --batch_size 64  \
        --patience 4 \ 
        --pretrained yes \
        --lr 5e-6
```
For PAWSX and XNLI you have to run train.py with the same parametrization<br>
To reproduce the results of the paper, we provide the commands with the training parameters in the file run.sh. The models were trained using NVIDIA 2080 Ti with 12 GB of GPU memory. 
