call .venv/Scripts/activate

python train_source_networks.py --dataset=first5_mnist &
python train_source_networks.py --dataset=last5_mnist &
python train_source_networks.py --dataset=mnist &
python train_source_networks.py --dataset=cifar10 --batch_size=128 &

python train_source_networks.py --dataset=mnist --arch=resnet18 --lr=0.1 &
python train_source_networks.py --dataset=cifar10 --arch=resnet18 --batch_size=128 --lr=0.1 --epochs 20 

REM python train_source_networks.py --dataset=first5_mnist --arch=lenet5_halfed &
REM python train_source_networks.py --dataset=last5_mnist --arch=lenet5_halfed &
REM python train_source_networks.py --dataset=mnist --arch=lenet5_halfed &
REM python train_source_networks.py --dataset=cifar10 --arch=lenet5_halfed --batch_size=128