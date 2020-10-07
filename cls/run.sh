python3 cls.py
python3 cls.py --model_path ../../model/flaubert1/flaubert1.model
python3 cls.py --model_path ../../model/dascim1/dascim2.model --batch_size 16
python3 cls.py --model_path ../../model/flaubert1/flaubert2.model --batch_size 16
python3 cls.py --model_path ../../model/cc.fr.300.bin --batch_size 32 --lr 5e-5 --algorithm fasttext
python3 cls.py --pretrained no --batch_size 16
python3 cls.py --dataset ../../datasets/CLS/music
python3 cls.py --model_path ../../model/flaubert1/flaubert1.model --dataset ../../datasets/CLS/music --batch_size 48
python3 cls.py --model_path ../../model/dascim1/dascim2.model --batch_size 16 --dataset ../../datasets/CLS/music --patience 6
python3 cls.py --model_path ../../model/flaubert1/flaubert2.model --batch_size 16 --dataset ../../datasets/CLS/music
python3 cls.py --model_path ../../model/cc.fr.300.bin --batch_size 8 --lr 5e-6 --dataset ../../datasets/CLS/music --algorithm fasttext
python3 cls.py --pretrained no --batch_size 16
python3 cls.py --dataset ../../datasets/CLS/dvd
python3 cls.py --model_path ../../model/flaubert1/flaubert1.model --dataset ../../datasets/CLS/dvd --batch_size 32
python3 cls.py --model_path ../../model/dascim1/dascim2.model --batch_size 16 --dataset ../../datasets/CLS/dvd
python3 cls.py --model_path ../../model/flaubert1/flaubert2.model --batch_size 16 --dataset ../../datasets/CLS/dvd
python3 cls.py --model_path ../../model/cc.fr.300.bin --batch_size 32 --lr 5e-5 --dataset ../../datasets/CLS/dvd --algorithm fasttext
python3 cls.py --pretrained no --batch_size 16 --dataset ../../datasets/CLS/dvd