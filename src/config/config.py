from dataclasses import dataclass

@dataclass
class ModelConfig:
    hidden_size: int = 768
    num_attention_heads: int = 12
    num_hidden_layers: int = 12
    patch_size: int = 16
    max_position_embeddings: int = 512
    num_modalities: int = 3  # text, image, video
    dropout_prob: float = 0.1
    attention_probs_dropout_prob: float = 0.1
    initializer_range: float = 0.02
    memory_size: int = 100000
    memory_dim: int = 768

class DataSourceConfig:
    WIKIPEDIA_SEEDS = [
        'https://en.wikipedia.org/wiki/Artificial_intelligence',
        'https://en.wikipedia.org/wiki/Machine_learning',
        'https://en.wikipedia.org/wiki/Deep_learning',
        'https://en.wikipedia.org/wiki/Computer_vision',
        'https://en.wikipedia.org/wiki/Natural_language_processing',
        'https://en.wikipedia.org/wiki/Robotics',
        'https://en.wikipedia.org/wiki/Neural_network',
        'https://en.wikipedia.org/wiki/Reinforcement_learning',
        'https://en.wikipedia.org/wiki/Computer_science',
        'https://en.wikipedia.org/wiki/Data_science'
    ]
    
    ARXIV_CATEGORIES = [
        'cs.AI', 'cs.LG', 'cs.CV', 'cs.CL', 'cs.RO', 'cs.NE',
        'stat.ML', 'cs.IR', 'cs.DC', 'cs.SI'
    ]
    
    IMAGE_DATASETS = [
        {'name': 'MNIST', 'loader': 'datasets.MNIST'},
        {'name': 'CIFAR10', 'loader': 'datasets.CIFAR10'},
        {'name': 'CIFAR100', 'loader': 'datasets.CIFAR100'},
        {'name': 'FashionMNIST', 'loader': 'datasets.FashionMNIST'}
    ]
    
    VIDEO_SOURCES = [
        'youtube.com', 'vimeo.com', 'dailymotion.com',
        'ted.com', 'coursera.org', 'udacity.com'
    ] 