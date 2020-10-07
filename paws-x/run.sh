python3 train.py
python3 train.py --model_path ../../model/flaubert1/flaubert1.model 
python3 train.py --model_path ../../model/dascim1/dascim2.model --lr 2e-4 --patience 4 --batch_size 16
python3 train.py --model_path ../../model/flaubert1/flaubert2.model 
python3 train.py --model_path ../../model/cc.fr.300.bin --algorithm fasttext --lr 2e-4 
python3 train.py --pretrained no