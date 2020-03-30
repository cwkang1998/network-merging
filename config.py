# Training settings
class Config:
    batch_size: int = 512
    test_batch_size: int = 1000
    epochs: int = 10
    lr: float = 0.01
    momentum: float = 0.9
    no_cuda: bool = False
    seed: int = 1
    log_interval: int = 10
    save_model: bool = False
    data_dir: str = "./cache/data/"
    output_dir: str = "./cache/models/"
