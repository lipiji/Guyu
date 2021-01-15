mkdir data
python prepare_data.py --src_file ./toy/toy.txt --tgt_file ./data/data.txt   --nprocessors 20

cd ./data
train="train.txt"
dev="dev.txt"
awk -v train="$train" -v dev="$dev" '{if(rand()<0.9) {print > train} else {print > dev}}' data.txt
