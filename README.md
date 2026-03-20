# BACF: Bayesian Attentional Collaborative Filtering

- 게재 일자: 2025.09.25.

- 수정 기간: 2026.07.

- 주 저자: [`Wang,J.`](https://github.com/jayarnim)

- 교신 저자: [`Lee,J.`](https://github.com/jaylee07)

## 개요

본 연구는 한국지능정보시스템학회 2024 추계학술대회에서 발표하였던 [`BAMF(Wang & Lee, 2024)`](https://github.com/jayarnim/2024-KIISS-BAMF) 의 후속 연구이다. 이전 연구에서는 다음의 두 가지를 확인하였다. 첫째, 조건부 선호 표현 도출 시 어느 엔티티의 과거 이력을 사용해야 하는가이다. 실험 결과, 두 엔티티의 이력을 모두 사용하였을 때 가장 우수한 성능을 보였다. 하지만 사용자 이력만을 사용하였을 때와 성능 격차가 두드러지지 않았다. 따라서 본 연구에서는 사용자 이력만을 사용하기로 결정하였다.

둘째, 전역 행동 표현이 조건부 선호 표현의 잔여 정보를 제공하는가, 만약 제공한다면 정보 합성 전략으로서 어떤 함수가 효과적인가이다. 실험 결과, 잔여 정보의 유효성을 확인할 수 없었다. 하지만 이는 단일 데이터 셋에서 확인한 결과이기 때문에, 정보량이 다른 데이터 환경에서는 결과가 달라질 수 있다고 판단하였다. 이에 본 연구에서는 조건부 선호 표현만 사용하지 않고 아래 합성 전략을 적용하여 각종 데이터 환경에서 잔여 정보의 유효성을 추가로 검증하였다.

- `sum`: 두 정보를 누적하여 합성하는 정보 누적 연산(보외법)
- `mean`: 두 정보를 균등 비율로 선택하여 합성하는 정보 선택 연산(보간법)
- `att`: 두 정보를 임의 비율로 선택하여 합성하는 정보 선택 연산(보간법)
- `prod`: 두 정보의 교집합을 부각하는 정보 여과 연산(상호작용)
- `cat`: 후속 레이어(e.g. mlp, linear, etc.)에서 합성하도록 두 정보를 보존함

본 연구에서는 mlp 기반 빈도적 추정 모형[^1][^2][^3] 대비 제안 모형이 비교 우위를 점하는 데이터 환경을 모색하였다. 이를 위하여 데이터의 신뢰성과 희소성 측면에서 차이가 있는 네 개의 데이터 셋을 사용하여 비교 모형과의 성능을 비교 검증하였다. 또한 이전 연구에서는 매칭 함수로서 gmf(general matrix factorization)만을 적용하였으나, 본 연구에서는 ncf(neural collaborative filtering)을 추가 적용하였다:

- amazon luxury beauty small 5-core (2018) [`link`](https://cseweb.ucsd.edu/~jmcauley/datasets/amazon_v2/)
- amazon digital music small 5-core (2014) [`link`](https://cseweb.ucsd.edu/~jmcauley/datasets/amazon/links.html)
- last.fm 2k (2011) [`link`](https://grouplens.org/datasets/hetrec-2011/)
- movielens latest small (2018) [`link`](https://grouplens.org/datasets/movielens/latest/)

실험 결과, 비교 모형 대비 제안 모형의 성능이 전반적으로 우수함을 확인하였다. 제안 모형의 비교 우위는 희소한 데이터 환경보다는, 신뢰도가 낮은 데이터 환경에서 두드러졌다. 이는 제안 모형이 본 연구의 문제의식에 적합한 접근법임을 시사한다. 데이터 희소성은 변분 분포의 분산 파라미터 설정과 관련이 있었다. 희소성이 높을수록 변분 분포의 분산 파라미터를 크게 설정하여 관측 정보에 기반한 의사 결정의 불확실성을 반영하였을 때 성능이 추가로 개선되었다.

한편, 전역 행동 표현의 잔여 정보 제공 여부는 데이터 밀집도에 따라 달리 나타났다. 밀집도가 높은 데이터 환경에서는 잔여 정보를 제공한다고 볼 수 없었으나, 밀집도가 낮은 경우에는 전역 행동 표현을 합성하였을 때 성능이 개선되었다. 단, 매칭 함수에 따라 효과적인 전략이 달랐다. 매칭 함수 학습을 병행하지 않을 때는 정보를 누적하는 가법적 합성 전략(`sum`)이, 병행할 때는 정보를 여과하는 쌍선형 사상 전략(`prod`)이 효과적이었다.

## 표기

### idx

- $u=0,1,2,\cdots,M-1$: target user
- $i=0,1,2,\cdots,N-1$: target item
- $j \in R_{u}^{+} \setminus \{i\}$: history items of target user (target item $i$ is excluded)

### vector

- $p \in \mathbb{R}^{M \times K}$: user id embedding vector (we define it as global behavior representation)
- $q \in \mathbb{R}^{N \times K}$: target item id embedding vector (we define it as global behavior representation)
- $h \in \mathbb{R}^{N \times K}$: history item id embedding vector
- $c_{u} \in \mathbb{R}^{M \times K}$: user context vector (we define it as conditional preference representation)
- $c_{i} \in \mathbb{R}^{N \times K}$: item context vector (we define it as conditional preference representation)
- $z_{u} \in \mathbb{R}^{M \times K}$: user refined representation
- $z_{i} \in \mathbb{R}^{N \times K}$: item refined representation
- $z_{u,i}$: $(u,i)$ pair predictive vector
- $x_{u,i}$: $(u,i)$ pair interaction logit
- $\hat{y}_{u,i}$: $(u,i)$ pair interaction probability

### function

- $\mathrm{mlp}(\cdot)$: multi-layer-perceptron
- $\mathrm{bam}(q,k,v)$: bayesian attention module (only single head)
- $\mathrm{comb}(\cdot)$: behavior rep & preference rep combination function (e.g. `sum`, `att`, `mean`, `prod`, `cat`)
- $\odot$: element-wise product
- $\oplus$: vector concatenation
- $\mathrm{ReLU}$: activation function, ReLU
- $\sigma$: activation function, sigmoid
- $W$: linear transformation matrix
- $h$: linear trainsformation vector
- $b$: bias term

## 모형

![01](/desc/model.png)

- global behavior:

$$\begin{aligned}
p_{u}
&=\mathrm{emb}(u)\\
q_{i}
&=\mathrm{emb}(i)
\end{aligned}$$

- conditional preference:

$$\begin{aligned}
c_{u}
&=\mathrm{bam}(p_{u}, h_{j}, h_{j}),\quad \forall j\in\mathcal{R}_{u}^{+}\setminus\{i\}\\
c_{i}
&=\mathrm{bam}(q_{i}, h_{j}, h_{j}),\quad \forall j\in\mathcal{R}_{u}^{+}\setminus\{i\}
\end{aligned}$$

- combination:

$$\begin{aligned}
z_{u}
&=\mathrm{comb}(p_{u}, c_{u})\\
z_{i}
&=\mathrm{comb}(q_{i}, c_{i})
\end{aligned}$$

- matching function:

$$\begin{aligned}
z_{u,i}
&=\begin{cases}
\mathrm{mlp}_{\mathrm{ReLU}}\left(\left[z_{u} \oplus z_{i}\right]\right)\\
z_{u} \odot z_{i}
\end{cases}
\end{aligned}$$

- prediction:

$$
\hat{y}_{u,i}=\sigma\left[h^{T}(W \cdot z_{u,i}+b)\right]
$$

- objective function:

$$
\mathcal{L}_{\mathrm{ELBO}}
:= \sum_{(u,i)\in\Omega}{\left(\mathrm{NLL} + \sum_{j \in R_{u}^{+} \setminus \{i\}}{\mathrm{KL}^{(u,j)}} + \sum_{j \in R_{u}^{+} \setminus \{i\}}{\mathrm{KL}^{(i,j)}} \right)}
$$

## 실험

성능을 측정하기 위하여 다음의 데이터 셋을 활용하였다:

- amazon luxury beauty small 5-core (2018) [`link`](https://cseweb.ucsd.edu/~jmcauley/datasets/amazon_v2/)
- amazon digital music small 5-core (2014) [`link`](https://cseweb.ucsd.edu/~jmcauley/datasets/amazon/links.html)
- last.fm 2k (2011) [`link`](https://grouplens.org/datasets/hetrec-2011/)
- movielens latest small (2018) [`link`](https://grouplens.org/datasets/movielens/latest/)

네 개 데이터 셋의 세부 정보는 다음의 표와 같다:

| | user | item | hist len (max) | (0.1) | (min) |interaction | density |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| a-beauty | $3,819$ | $1,581$ | $127$ | $15$ | $4$ | $34,278$ | $0.0057$ |
| a-music | $5,541$ | $3,568$ | $528$ | $20$ | $5$ | $64,706$ | $0.0033$ |
| last.fm | $1,892$ | $17,632$ | $50$ | $50$ | $1$ | $92,834$ | $0.0028$ |
| movielens | $610$ | $9,724$ | $2,698$ | $400$ | $20$ | $100,836$ | $0.0170$ |

`a-beauty`, `a-music`, `movielens` 는 모두 명시적 피드백을 제공하는 데이터 셋이다. 본 연구에서는 관측이 선호 신호로서 모호한 데이터 환경을 논하고 있으므로 $1-5$ 점을 모두 관측으로 간주하였다. 따라서 세 개의 데이터 셋은 모두 관측값의 신뢰도가 낮은 환경이 된다. `last.fm` 은 관측 누적 횟수를 제공한다. 이는 암시적 피드백이긴 하나, 제1사분위수 $107$ 회, 제2사분위수 $260$ 회, 제3사분위수 $614$ 회, 최댓값 $352,698$ 회로 관측값의 신뢰도가 높다. 한편, 밀집도 측면에서는 `movielens` 가 가장 조밀하고, `a-beauty`, `a-music`, `last.fm` 는 희소하다.

데이터 셋을 `trn`, `val`, `tst` 각각 8:1:1 로 사용자 기준 계층적 분할하였다. `opt`(`trn`) 데이터 셋에 대하여 1:4, `msr`(`val`, `tst`) 데이터 셋에 대하여 1:99 비율로 네거티브 샘플링을 적용하였다. 손실 함수로는 bce(binary cross entropy)를 적용하여 훈련하였고, 조기종료 지표로는 `ndcg@10` 을 적용하여 모니터링하였다. 초기 10 epoch 에 대해서는 조기종료 여부를 모니터링하지 않았고, 11 epoch 부터 모니터링하여 지표가 최대 5회 개선되지 않을 경우 학습을 조기종료하였다.

단, nll(negative log likelihood)의 스케일은 $0.5$ 이하인 것에 반해, kld(kullback–leibler divergence)의 스케일은 약 $2.0$ 수준에서 형성되어 있다. 두 항의 스케일이 맞지 않으므로 균형을 맞추기 위하여 $\beta$ 값을 $0.1$ 로 설정하였다. 또한, 학습 초기에 kld 항의 비중이 클 경우 사전 정보만 반영하고 관측값 정보를 반영하기 어려울 수 있다. 이에 초기 10 epoch 동안 선형 어닐링(linear annealing)을 적용하여 관측값 정보를 반영하고, 이후에 사전 정보를 참조하여 과적합을 방지하도록 하였다.

한편, 제안 모형은 임베딩 생성 단계에서 엔티티의 과거 이력을 집계한다. `movielens` 에서는 일부 사용자의 상호작용 이력이 $2,000$ 건이 넘는데 반해, 상위 $0.1$ 사용자의 이력은 $400$ 건이다. 효율성을 도모하기 위하여 선별 점수 기준 상위 $400$ 건의 이력만을 활용하였다. 선별 점수로는 상호작용 빈도와 TF-IDF 를 활용하였으며, TF-IDF 를 활용하였을 때 추가적인 성능 개선이 있었다. TF-IDF 를 활용할 때는 문서를 상호작용 이력에, 단어(혹은 토큰)를 해당 이력의 구성자에 대응하여 점수를 산출하였다.

## 결과

실험 결과, 비교 모형 대비 제안 모형의 성능이 전반적으로 우수함을 확인하였다. 단, 그 격차는 데이터 환경에 따라 차이가 있었다. 개별 관측치의 신뢰도가 낮은 데이터 환경에서는 제안 모형의 비교 우위가 두드러진 반면, 밀집도가 낮더라도 신뢰도가 높은 데이터 환경에서는 비교 우위가 뚜렷하지 않았다. **이는 제안 모형의 개선 효과가 데이터의 희소성보다, 개별 신호가 선호 신호로서 얼마나 신뢰할 수 있는지에 더 크게 의존한다는 점을 시사한다.**

**데이터 희소성은 변분 분포의 분산 파라미터 설정과 관련이 있었다.** 희소성이 높은 데이터 환경에서는 변분 분포의 분산을 크게 설정하였을 때 성능이 개선되었다. 이는 희소한 데이터 환경에서는 관측 정보에 기반한 판단의 불확실성이 크다는 점에서 기인한다. 때문에 가능한 여러 선호 가설을 유지하는 것이 효과적이다. 하지만 관측 정보가 충분한 환경에서는, 반대로 변분 분포를 좁게 유지하여 관측 정보에 기반한 의사 결정을 신뢰하는 것이 유리했다.

한편, **전역 행동 표현의 잔여 정보 제공 여부는 데이터 밀집도에 따라 달리 나타났다.** 밀집도가 높은 데이터 환경에서는 조건부 선호 표현에 전역 행동 표현을 합성하였을 때 성능이 감소하였으나, 밀집도가 낮은 경우에는 반대로 합성하였을 때 성능이 개선되었다. 밀집도가 높을 때는 과거 이력이 풍부하여 이로부터 유효 정보를 충분히 제공 받을 수 있다. 때문에 전역 행동 표현이 잔여 정보를 제공한다고 보기 어렵다. 하지만 밀집도가 낮을 때는 과거 이력이 충분히 확보되지 않기 때문에, 전역 행동 표현으로 부족한 정보를 보충하는 것이 유리하다.

단, 매칭 함수에 따라 효과적인 전략이 달랐다. 이는 매칭 함수의 학습 가능 여부에 기인한다. gmf(general matrix factorization)를 적용할 때는 매칭 단계에서 정보를 재구성할 수 없다. 때문에 가능한 많은 정보를 보존하는 가법적 합성 전략(`sum`)이 효과적이었다. 이와 달리, ncf(neural collaborative filtering)를 적용할 경우 매칭 함수 학습이 수반된다. 때문에 후속 단계에서 정보를 재구성하기 용이하도록 정보를 선별하여 전달하는 쌍선형 사상 전략(`prod`)이 효과적이었다.

## 비교 모형[^4]

- [`ncf`](https://github.com/jayarnim/RS-NCF) He, X., Liao, L., Zhang, H., Nie, L., Hu, X., & Chua, T. S. (2017, April). Neural collaborative filtering. In Proceedings of the 26th international conference on world wide web (pp. 173-182).

- [`dmf`](https://github.com/jayarnim/RS-DMF) Xue, H. J., Dai, X., Zhang, J., Huang, S., & Chen, J. (2017, August). Deep matrix factorization models for recommender systems. In IJCAI (Vol. 17, pp. 3203-3209).

- [`deepcf`](https://github.com/jayarnim/RS-DeepCF) Deng, Z. H., Huang, L., Wang, C. D., Lai, J. H., & Yu, P. S. (2019, July). Deepcf: A unified framework of representation learning and matching function learning in recommender system. In Proceedings of the AAAI conference on artificial intelligence (Vol. 33, No. 01, pp. 61-68).

- [`j-ncf`](https://github.com/jayarnim/RS-J-NCF) Chen, W., Cai, F., Chen, H., & Rijke, M. D. (2019). Joint neural collaborative filtering for recommender systems. ACM Transactions on Information Systems (TOIS), 37(4), 1-30.

- [`convncf`](https://github.com/jayarnim/RS-ConvNCF) He, X., Du, X., Wang, X., Tian, F., Tang, J., & Chua, T. S. (2018). Outer product-based neural collaborative filtering. arXiv preprint arXiv:1808.03912.

- [`comet`](https://github.com/jayarnim/RS-COMET) Lin, Z., Feng, L., Guo, X., Zhang, Y., Yin, R., Kwoh, C. K., & Xu, C. (2023). Comet: Convolutional dimension interaction for collaborative filtering. ACM Transactions on Intelligent Systems and Technology, 14(4), 1-18.

- [`dacr`](https://github.com/jayarnim/RS-DACR) Cui, C., Qin, J., & Ren, Q. (2022). Deep collaborative recommendation algorithm based on attention mechanism. Applied Sciences, 12(20), 10594.

- [`drnet`](https://github.com/jayarnim/RS-DRNet) Ji, D., Xiang, Z., & Li, Y. (2020). Dual relations network for collaborative filtering. IEEE Access, 8, 109747-109757.

- [`delf`](https://github.com/jayarnim/RS-DELF) Cheng, W., Shen, Y., Zhu, Y., & Huang, L. (2018, July). DELF: A dual-embedding based deep latent factor model for recommendation. In IJCAI (Vol. 18, pp. 3329-3335).

- [`fism`](https://github.com/jayarnim/RS-FISM) Kabbur, S., Ning, X., & Karypis, G. (2013, August). Fism: factored item similarity models for top-n recommender systems. In Proceedings of the 19th ACM SIGKDD international conference on Knowledge discovery and data mining (pp. 659-667).

- [`nais`](https://github.com/jayarnim/RS-NAIS) He, X., He, Z., Song, J., Liu, Z., Jiang, Y. G., & Chua, T. S. (2018). NAIS: Neural attentive item similarity model for recommendation. IEEE Transactions on Knowledge and Data Engineering, 30(12), 2354-2366.

## 성능 지표 요약[^5]

### a-beauty

- ncf:

    | Comb |        HR@5 |       HR@10 |      NDCG@5 |     NDCG@10 |
    | ---- | ----------: | ----------: | ----------: | ----------: |
    | att  |     $0.580$ |     $0.658$ |     $0.548$ |     $0.573$ |
    | cat  |     $0.592$ |     $0.671$ |     $0.556$ |     $0.581$ |
    | mean |     $0.586$ |     $0.661$ |     $0.554$ |     $0.577$ |
    | none |     $0.611$ |     $0.677$ | **$0.576$** | **$0.596$** |
    | prod | **$0.622$** | **$0.686$** |     $0.574$ |     $0.594$ |
    | sum  |     $0.585$ |     $0.664$ |     $0.550$ |     $0.575$ |

- gmf:

    | Comb |        HR@5 |       HR@10 |      NDCG@5 |     NDCG@10 |
    | ---- | ----------: | ----------: | ----------: | ----------: |
    | att  |     $0.595$ |     $0.668$ |     $0.559$ |     $0.582$ |
    | cat  | **$0.624$** | **$0.694$** | **$0.589$** | **$0.611$** |
    | mean |     $0.619$ |     $0.693$ |     $0.577$ |     $0.601$ |
    | none |     $0.616$ |     $0.684$ |     $0.574$ |     $0.595$ |
    | prod |     $0.609$ |     $0.685$ |     $0.566$ |     $0.590$ |
    | sum  |     $0.622$ |     $0.693$ |     $0.586$ |     $0.608$ |

### a-music

- ncf:

    | Comb |        HR@5 |       HR@10 |      NDCG@5 |     NDCG@10 |
    | ---- | ----------: | ----------: | ----------: | ----------: |
    | att  |     $0.577$ |     $0.719$ |     $0.401$ |     $0.447$ |
    | cat  |     $0.577$ |     $0.714$ |     $0.392$ |     $0.438$ |
    | mean |     $0.565$ |     $0.706$ |     $0.391$ |     $0.438$ |
    | none |     $0.595$ | **$0.732$** |     $0.416$ |     $0.463$ |
    | prod | **$0.611$** |     $0.731$ | **$0.436$** | **$0.477$** |
    | sum  |     $0.570$ |     $0.712$ |     $0.393$ |     $0.440$ |

- gmf:

    | Comb |        HR@5 |       HR@10 |      NDCG@5 |     NDCG@10 |
    | ---- | ----------: | ----------: | ----------: | ----------: |
    | att  |     $0.615$ |     $0.745$ |     $0.437$ |     $0.482$ |
    | cat  |     $0.602$ |     $0.736$ |     $0.423$ |     $0.469$ |
    | mean |     $0.626$ |     $0.754$ |     $0.444$ |     $0.489$ |
    | none |     $0.613$ |     $0.754$ |     $0.424$ |     $0.472$ |
    | prod |     $0.581$ |     $0.706$ |     $0.401$ |     $0.443$ |
    | sum  | **$0.633$** | **$0.766$** | **$0.454$** | **$0.499$** |

### last.fm

- ncf:

    | Comb |        HR@5 |       HR@10 |      NDCG@5 |     NDCG@10 |
    | ---- | ----------: | ----------: | ----------: | ----------: |
    | att  |     $0.884$ |     $0.944$ |     $0.449$ |     $0.526$ |
    | cat  |     $0.895$ |     $0.946$ |     $0.454$ |     $0.534$ |
    | mean |     $0.903$ |     $0.949$ |     $0.447$ |     $0.530$ |
    | none |     $0.902$ |     $0.948$ |     $0.479$ |     $0.557$ |
    | prod | **$0.911$** | **$0.950$** | **$0.484$** | **$0.563$** |
    | sum  |     $0.876$ |     $0.941$ |     $0.435$ |     $0.516$ |

- gmf:

    | Comb |        HR@5 |       HR@10 |      NDCG@5 |     NDCG@10 |
    | ---- | ----------: | ----------: | ----------: | ----------: |
    | att  |     $0.916$ |     $0.957$ |     $0.482$ |     $0.566$ |
    | cat  |     $0.920$ |     $0.960$ |     $0.484$ |     $0.564$ |
    | mean |     $0.916$ |     $0.958$ |     $0.478$ |     $0.557$ |
    | none |     $0.908$ |     $0.955$ |     $0.475$ |     $0.555$ |
    | prod |     $0.910$ |     $0.947$ |     $0.472$ |     $0.544$ |
    | sum  | **$0.925$** | **$0.963$** | **$0.491$** | **$0.570$** |

### movielens

- ncf:

    | Comb |        HR@5 |       HR@10 |      NDCG@5 |     NDCG@10 |
    | ---- | ----------: | ----------: | ----------: | ----------: |
    | att  |     $0.869$ |     $0.948$ |     $0.491$ |     $0.499$ |
    | cat  |     $0.880$ |     $0.946$ |     $0.484$ |     $0.495$ |
    | mean | **$0.887$** |     $0.954$ |     $0.504$ |     $0.514$ |
    | none |     $0.879$ | **$0.957$** | **$0.529$** | **$0.526$** |
    | prod |     $0.882$ |     $0.951$ |     $0.499$ |     $0.513$ |
    | sum  |     $0.862$ |     $0.949$ |     $0.507$ |     $0.512$ |

- gmf:

    | Comb |        HR@5 |       HR@10 |      NDCG@5 |     NDCG@10 |
    | ---- | ----------: | ----------: | ----------: | ----------: |
    | att  |     $0.884$ |     $0.957$ |     $0.523$ |     $0.536$ |
    | cat  |     $0.879$ |     $0.951$ |     $0.523$ |     $0.526$ |
    | mean |     $0.890$ |     $0.951$ |     $0.534$ |     $0.540$ |
    | none | **$0.898$** |     $0.949$ | **$0.547$** | **$0.550$** |
    | prod |     $0.875$ |     $0.944$ |     $0.505$ |     $0.512$ |
    | sum  |     $0.892$ | **$0.959$** |     $0.530$ |     $0.541$ |

## Reference

- 이영의. (2015). 베이즈주의: 합리성으로부터 객관성으로의 여정. 한국문화사

- 고지마 히로유키. (2017). 세상에서 가장 쉬운 베이즈 통계학 입문. 지성사

- Jannach, D., Lerche, L., & Zanker, M. (2018). Recommending based on implicit feedback. In Social information access: systems and technologies (pp. 510-569). Cham: Springer International Publishing.

- He, X., Liao, L., Zhang, H., Nie, L., Hu, X., & Chua, T. S. (2017, April). Neural collaborative filtering. In Proceedings of the 26th international conference on world wide web (pp. 173-182).

- He, X., He, Z., Song, J., Liu, Z., Jiang, Y. G., & Chua, T. S. (2018). NAIS: Neural attentive item similarity model for recommendation. IEEE Transactions on Knowledge and Data Engineering, 30(12), 2354-2366.

- Fan, X., Zhang, S., Chen, B., & Zhou, M. (2020). Bayesian attention modules. Advances in Neural Information Processing Systems, 33, 16362-16376.

[^1]: 결정론(determinism)은 인과관계에 우연이 없다는 입장으로, 그 대척점에 있는 입장은 자유의지론(free will)에 해당한다. 확률론(probability theory)은 모수(parameter)가 결정되어 있음을 부정하지 않는다. 다만 이 모수의 참값을 추정하기에 불확실한 요소들을 인정하고 반영할 뿐이다. 예컨대 표집의 무작위성(aleatoric uncertainty)이나 연구자의 불완전한 지식 체계(epistemic uncertainty) 등이 이에 해당한다.

[^2]: 단, 빈도주의(frequentist)는 모수 자체를 규명하고자 하므로, 이를 미지의 상수(constant)로 설정하고 우도를 최대화하는 값으로 추정한다(max. likelihood estimation). 반면에 베이지안(bayesianism)은 모수에 대한 연구자의 인식 상태를 규명하고자 하므로, 모수를 확률 변수(random variable)로 설정하고 사전 정보와 관측값에 근거하여 그 사후 분포(posterior)를 추정한다.

[^3]: 인공신경망은 구조상 모수(parameter)와 입력값이 주어지면 출력이 결정된다는 점에서 결정론적(determinism) 모형에 해당한다. 하지만 모수를 미지의 상수(constant)로 설정하고, 우도를 최대화하는 값으로 추정한다(max. likelihood estimation). 따라서 통계적 추정의 관점에서는 빈도주의 접근(frequentist)과 구분되지 않는다. 따라서 본 연구에서는 논지 전개 시 결정론을 빈도주의와 별개로 구분하지 않겠다.

[^4]: 비교 모형으로는 gmf(general matrix factorization) 혹은 ncf(neural collaborative filtering) 기반 잠재요인 모형을 사용하였다.

[^5]: 본 레파지토리에서는 분산 파라미터를 변분 분포 $0.1$, 사전 분포 $1.0$ 으로 고정하여 실험을 수행하였다. 이하는 해당 파라미터 값으로 고정하였을 때 산출한 성능 값이다. `movielens` 에서는 변분 분포 $0.1-0.3$, `last.fm` 의 경우 $0.2-0.3$, `a-beauty`, `a-music` 에서는 $0.2-0.4$ 사이에서 성능이 추가 향상되었다.