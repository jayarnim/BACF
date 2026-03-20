import torch
import torch.nn as nn
from components.histories import Histories
from components.base import BayesModel
from components.base import BayesModelOutput
from .layers.embedding import build  as build_embedding_layer
from .layers.bam import BayesianAttentionModules
from .layers.combination import build as build_comb_layer
from .layers.matching import build as build_matching_layer
from .layers.prediction import ProjectionLayer


class BayesianAttentionalMatrixFactorization(BayesModel):
    def __init__(
        self,
        histories: Histories,
        num_users: int,
        num_items: int,
        embedding_dim: int,
        score: str,
        sampler: str,
        param_q: float,
        param_p: float,
        beta: float,
        comb: str=None,
        dropout: float=None,
    ):
        super().__init__(locals())

        self.histories = histories

        # EMBEDDINGS ==========
        components = dict(
            target=build_embedding_layer(
                name="idx",
                num_users=num_users,
                num_items=num_items,
                embedding_dim=embedding_dim,
            ),
            history=nn.Embedding(
                num_embeddings=num_items+2, 
                embedding_dim=embedding_dim,
                padding_idx=0,
            ),
        )
        self.embedding = nn.ModuleDict(components)

        # POOLING ==========
        kwargs = dict(
            score=score,
            sampler=sampler,
            dim=embedding_dim, 
            param_q=param_q,
            param_p=param_p,
            beta=beta,
            dropout=dropout,
        )
        components = dict(
            user=BayesianAttentionModules(**kwargs),
            item=BayesianAttentionModules(**kwargs),
        )
        self.pooling = nn.ModuleDict(components)

        # COMBINATION ==========
        if comb:
            kwargs = dict(
                name=comb,
                dim=embedding_dim,
            )
            components = dict(
                user=build_comb_layer(**kwargs),
                item=build_comb_layer(**kwargs),
            )
            self.comb = nn.ModuleDict(components)
        else:
            self.comb = None

        # MATCHING ==========
        self.matching = build_matching_layer(
            name="mf",
        )

        # PREDICTION ==========
        self.prediction = ProjectionLayer(
            dim=(
                embedding_dim*2
                if comb=="cat"
                else embedding_dim
            ),
        )

        self.init_embeddings()

    def forward(
        self, 
        user_idx: torch.Tensor, 
        item_idx: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # HIST IDX ==========
        hist_idx, mask = self.histories(user_idx, item_idx)

        # EMBEDDING ==========
        user_emb, item_emb = self.embedding["target"](user_idx, item_idx)
        hist_emb = self.embedding["history"](hist_idx)

        # POOLING HISTORIES ==========
        user_output = self.pooling["user"](
            q=user_emb,
            k=hist_emb,
            v=hist_emb,
            mask=mask,
        )
        item_output = self.pooling["item"](
            q=item_emb,
            k=hist_emb,
            v=hist_emb,
            mask=mask,
        )

        # COMBINATION ==========
        if self.comb:
            user_combined = self.comb["user"](user_emb, user_output.context)
            item_combined = self.comb["item"](item_emb, item_output.context)
        else:
            user_combined = user_output.context
            item_combined = item_output.context

        # MATCHING ==========
        X_pred = self.matching(user_combined, item_combined)

        return X_pred, ( user_output.kld + item_output.kld ) / 2

    def predict(
        self, 
        user_idx: torch.Tensor, 
        item_idx: torch.Tensor,
    ) -> BayesModelOutput:
        X_pred, kld = self.forward(user_idx, item_idx)
        logit = self.prediction(X_pred).squeeze(-1)
        return BayesModelOutput(
            logit=logit, 
            kld=kld,
        )

    def init_embeddings(self):
        kwargs = dict(
            tensor=self.embedding["history"].weight, 
            mean=0.0, 
            std=0.01,
        )
        nn.init.normal_(**kwargs)