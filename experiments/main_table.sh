# Reproduce results of the main table in the paper for each dataset

python main.py --dataset cornell --epochs 300 --nlayers 1 --verbose False --model GCNH --dropout 0.25 --nhid 16 --batch_size 50
python main.py --dataset texas --epochs 300 --nlayers 1 --verbose False --model GCNH --dropout 0.25 --nhid 32 --batch_size 200
python main.py --dataset wisconsin --epochs 300 --nlayers 2 --verbose False --model GCNH --dropout 0.3 --nhid 32 --batch_size 50
python main.py --dataset film --epochs 150 --nlayers 2 --verbose False --model GCNH --dropout 0.6 --nhid 32 --batch_size 500
python main.py --dataset chameleon --epochs 1000 --nlayers 1 --verbose False --model GCNH --dropout 0.0 --nhid 32 --batch_size 300
python main.py --dataset squirrel --epochs 1500 --nlayers 1 --verbose False --model GCNH --dropout 0.0 --nhid 32 --batch_size 1400
python main.py --dataset cora --epochs 300 --nlayers 2 --verbose False --model GCNH --dropout 0.75 --nhid 64 --batch_size 150
python main.py --dataset citeseer --epochs 300 --nlayers 1 --verbose False --model GCNH --dropout 0.25 --nhid 16 --batch_size 300