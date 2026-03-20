from ..config.model import (
    BACFCfg,
)


def auto(cfg):
    model = cfg["model"]["name"]
    if model=="bacf":
        return bacf(cfg)
    else:
        raise ValueError("invalid model name in .yaml config")


def bacf(cfg):
    return BACFCfg(
        num_users=cfg["data"]["entity"]["num_users"],
        num_items=cfg["data"]["entity"]["num_items"],
        embedding_dim=cfg["model"]["embedding_dim"],
        hidden_dim=cfg["model"]["hidden_dim"],
        dist=cfg["model"]["dist"],
        hyper_approx=cfg["model"]["hyper_approx"],
        hyper_prior=cfg["model"]["hyper_prior"],
        beta=cfg["model"]["beta"],
        dropout=cfg["model"]["dropout"],
        comb=cfg["model"]["comb"],
    )