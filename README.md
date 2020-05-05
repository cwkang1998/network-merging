# Towards Neural Network Merging

All the experiments would be ran through 10 distinct random seed. As such, do expect that the application would 
take some time to finish. 

Another thing to note is that the experiments were ran on a windows machine, using the CPU instead of GPU, some 
problems may be encountered if not running under such environment, and may require some modification of the code 
to account for that.

## Requirements 

The software is running with the current libraries, do remember to install them before running.

- Python 3.7 and above
- pytorch 1.4.0
- torchvision 0.5.0
- imgaug 0.2.6
- Pillow 7.0.0

## Training base networks

Training for the base networks can be invoked using the command:

```bash
python train_source_networks.py --dataset=<dataset> --batch_size=<batch size>
```

For the experiments on MNIST and CIFAR10 on both the problems, the following command was used:

```bash
python train_source_networks.py --dataset=first5_mnist
python train_source_networks.py --dataset=last5_mnist
python train_source_networks.py --dataset=mnist
python train_source_networks.py --dataset=cifar10 --batch_size=128
```

For the full list of arguments please do look into the [source file](./train_source_networks.py).


## Training PAN

Training for the pattern attribution networks(PAN) for the base networks can be invoked using the command:

```bash
python train_pan.py --dataset=<problem> --pan_type=<logits, features, agnostic_logits>
```

Note that this should only be ran after the base network has already been trained.

For the experiments on both the problems, the following command was used:

```bash
python train_pan.py --dataset=disjoint_mnist --pan_type=logits
python train_pan.py --dataset=disjoint_mnist --pan_type=feature
python train_pan.py --dataset=disjoint_mnist --pan_type=agnostic_logits

python train_pan.py --dataset=mnist_cifar10 --pan_type=logits
python train_pan.py --dataset=mnist_cifar10 --pan_type=feature
python train_pan.py --dataset=mnist_cifar10 --pan_type=agnostic_logits
```

Here, agnostic logits refers to the logits based activation statistics based pan.

For the full list of arguments please do look into the [source file](./train_pan.py).

## Running the experiments

The experiments can be run like so:

```bash
python merge.py --dataset=<problem> --arch=lenet5 --experiment=<experiment types>
```

Note that the prerequisite for running the experiments must all already exists and prepared.

For the experiments on both the problems, the following command was used:

```bash
call .venv/Scripts/activate

python merge.py --dataset=disjoint_mnist --arch=lenet5 --experiment=logits_statistics
python merge.py --dataset=mnist_cifar10 --arch=lenet5 --experiment=logits_statistics

python merge.py --dataset=disjoint_mnist --arch=lenet5 --experiment=multi_pass_aug_mean
python merge.py --dataset=mnist_cifar10 --arch=lenet5 --experiment=multi_pass_aug_mean

python merge.py --dataset=disjoint_mnist --arch=lenet5 --experiment=multi_pass_aug_voting
python merge.py --dataset=mnist_cifar10 --arch=lenet5 --experiment=multi_pass_aug_voting

# Logits based pan
python merge.py --dataset=disjoint_mnist --arch=lenet5 --experiment=smart_coord --pan_type=logits
python merge.py --dataset=mnist_cifar10 --arch=lenet5 --experiment=smart_coord --pan_type=logits

# Feature based pan
python merge.py --dataset=disjoint_mnist --arch=lenet5 --experiment=smart_coord --pan_type=feature
python merge.py --dataset=mnist_cifar10 --arch=lenet5 --experiment=smart_coord --pan_type=feature 

# logits activation statistics based pan
python merge.py --dataset=disjoint_mnist --arch=lenet5 --experiment=smart_coord --pan_type=agnostic_logits
python merge.py --dataset=mnist_cifar10 --arch=lenet5 --experiment=smart_coord --pan_type=agnostic_logits
```

For the full list of arguments please do look into the [source file](./merge.py).