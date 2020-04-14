call .venv/Scripts/activate

python train_source_networks.py --dataset first5_mnist &
python train_source_networks.py --dataset last5_mnist &
python train_source_networks.py --dataset mnist &
python train_source_networks.py --dataset cifar10 &

python train_source_networks.py --dataset first5_mnist --arch lenet5_halfed &
python train_source_networks.py --dataset last5_mnist --arch lenet5_halfed &
python train_source_networks.py --dataset mnist --arch lenet5_halfed &
python train_source_networks.py --dataset cifar10 --arch lenet5_halfed