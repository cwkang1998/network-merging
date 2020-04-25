call .venv/Scripts/activate

python train_pan.py --dataset=disjoint_mnist --pan_type=logits &
python train_pan.py --dataset=disjoint_mnist --pan_type=feature &
python train_pan.py --dataset=disjoint_mnist --pan_type=agnostic_logits &

python train_pan.py --dataset=mnist_cifar10 --pan_type=logits &
python train_pan.py --dataset=mnist_cifar10 --pan_type=feature &
python train_pan.py --dataset=mnist_cifar10 --pan_type=agnostic_logits

REM python train_pan.py --dataset=mnist_cifar10 --pan_type=logits --arch=resnet18 &
REM python train_pan.py --dataset=mnist_cifar10 --pan_type=feature --arch=resnet18