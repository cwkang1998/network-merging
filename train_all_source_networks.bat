call .venv/Scripts/activate

start python train_source_networks.py --dataset first5_mnist
start python train_source_networks.py --dataset last5_mnist
start python train_source_networks.py --dataset mnist
start python train_source_networks.py --dataset cifar10

start python train_source_networks.py --dataset first5_mnist --arch lenet5_halfed
start python train_source_networks.py --dataset last5_mnist --arch lenet5_halfed
start python train_source_networks.py --dataset mnist --arch lenet5_halfed
start python train_source_networks.py --dataset cifar10 --arch lenet5_halfed