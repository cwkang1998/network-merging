call .venv/Scripts/activate

python merge.py --dataset=disjoint_mnist --arch=lenet5 --experiment=logits_statistics &
python merge.py --dataset=mnist_cifar10 --arch=lenet5 --experiment=logits_statistics & 
REM python merge.py --dataset=mnist_cifar10 --arch=resnet18 --experiment=logits_statistics &

python merge.py --dataset=disjoint_mnist --arch=lenet5 --experiment=multi_pass_aug_mean &
python merge.py --dataset=mnist_cifar10 --arch=lenet5 --experiment=multi_pass_aug_mean & 
REM python merge.py --dataset=mnist_cifar10 --arch=resnet18 --experiment=multi_pass_aug_mean &

python merge.py --dataset=disjoint_mnist --arch=lenet5 --experiment=multi_pass_aug_voting &
python merge.py --dataset=mnist_cifar10 --arch=lenet5 --experiment=multi_pass_aug_voting & 
REM python merge.py --dataset=mnist_cifar10 --arch=resnet18 --experiment=multi_pass_aug_voting

REM Logits based pan
python merge.py --dataset=disjoint_mnist --arch=lenet5 --experiment=smart_coord --pan_type=logits &
python merge.py --dataset=mnist_cifar10 --arch=lenet5 --experiment=smart_coord --pan_type=logits & 
REM python merge.py --dataset=mnist_cifar10 --arch=resnet18 --experiment=smart_coord --pan_type=logits &

REM Feature based pan
python merge.py --dataset=disjoint_mnist --arch=lenet5 --experiment=smart_coord --pan_type=feature &
python merge.py --dataset=mnist_cifar10 --arch=lenet5 --experiment=smart_coord --pan_type=feature & 
REM python merge.py --dataset=mnist_cifar10 --arch=resnet18 --experiment=smart_coord --pan_type=feature

REM agnostic logits based pan
python merge.py --dataset=disjoint_mnist --arch=lenet5 --experiment=smart_coord --pan_type=agnostic_logits &
python merge.py --dataset=mnist_cifar10 --arch=lenet5 --experiment=smart_coord --pan_type=agnostic_logits