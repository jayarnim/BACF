from dataclasses import dataclass


@dataclass
class BACFCfg:
    num_users: int
    num_items: int
    embedding_dim: int
    hidden_dim: list
    dist: str
    hyper_approx: float
    hyper_prior: float
    beta: float
    dropout: float
    comb: str