from typing import List
# Training settings
class Config:
    batch_size: int = 512
    test_batch_size: int = 1000
    epochs: int = 10
    lr: float = 0.01
    momentum: float = 0.9
    no_cuda: bool = False
    seeds: List = [770597, 600522, 416719, 306890, 201746, 156886, 929271, 247945, 5028, 436287]
    log_interval: int = 10
    save_model: bool = False
    data_dir: str = "./cache/data/"
    output_dir: str = "./cache/models/"