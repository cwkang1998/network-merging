call .venv/Scripts/activate

python merge.py --dataset=disjoint_mnist --arch=lenet5 --experiment=logits_statistics &
python merge.py --dataset=mnist_cifar10 --arch=lenet5 --experiment=logits_statistics &

python merge.py --dataset=disjoint_mnist --arch=lenet5_halfed --experiment=logits_statistics &
python merge.py --dataset=mnist_cifar10 --arch=lenet5_halfed --experiment=logits_statistics &
