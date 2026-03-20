# Bayesian Attentional Collaborative Filtering

- 게재일자: 2025.9.25.

- 제1저자: [`Wang,J.`](https://github.com/jayarnim)

- 교신저자: [`Lee,J.`](https://github.com/jaylee07)

## 개요

본 연구는 한국지능정보시스템학회 2024 추계학술대회에서 발표된 [`BAMF(Wang & Lee, 2024)`](https://github.com/jayarnim/BAMF)의 후속 확장 연구이다. 모형 설계 상의 차이점은 매칭 함수를 Matrix Factorization 에서 Neural Collaborative Filtering 으로 교체했다는 점이다. 또한 초기 연구(BAMF)의 경우 단일 데이터 셋만을 활용하여 성능을 검증하였으나, 본 연구에서는 데이터의 신뢰성과 희소성 측면에서 차이가 있는 네 개의 데이터 셋을 활용하여 제안 모형의 장점이 부각되는 환경을 모색하였다. 그 결과, 상호작용 이력이 적정 수준 확보되면서 데이터 희소성이 높고 개별 관측치가 선호 신호로서 불확실한 환경에서 다른 잠재요인 모형들에 비해 성능이 우수함을 검증하였다. OTT 서비스와 같이 신규 아이템이 자주 출시되는 환경이 그 일례가 될 수 있다. 따라서, 비록 초기 연구에서는 두 엔티티의 상호작용 이력을 모두 활용하였을 때 성능이 극대화되었으나, 본 연구에서는 사용자 이력만을 활용하였다. (이하 내용은 초기 연구의 레파지토리 `README.md` 와 상당 부분 중복될 수 있다.)

## 아키텍처

![01](/desc/model.png)

제안 모형에서 엔티티의 잠재 표현은 두 가지 용도로 사용되고 있다. 매칭 대상일 때의 잠재 표현은 사용자-아이템 상호작용에서 관측되는 행동 신호를 나타내고, 상호작용 이력일 때의 잠재 표현은 엔티티의 선호를 형성하는 맥락 정보를 나타낸다. 논문에서는 임베딩 행렬을 용도에 따라 각각 생성하지 않고, 하나의 임베딩 행렬을 선형 변환하고 있다. 이는 행동 신호와 맥락 정보가 동일한 잠재요인에 기반하지 아니하고 서로 다른 기저 공간에서 해석되어야 함을 전제한다. 하지만 본 레파지토리에서는 용도에 따라 임베딩 행렬을 각각 생성하는 방향으로 조정하였다. 이는 행동 표현과 선호 맥락이 동일한 기저를 공유하면서 제공하는 정보가 다른 표현임을 전제한다.

한편, 제안 모형은 행동 표현과 선호 표현을 어떻게 결합할 것인지에 따라 구성 전략이 세분화될 수 있다. 행동 표현과 선호 표현의 결합 전략과 선형적 의미는 다음과 같다. 아래 전략들은 DNCF(He et al., 2021)에서 제안된 함수들이다. 초기 연구와 달리, `prod` 를 적용하였을 때 가장 높은 성능을 보였다.

- `sum`: 두 벡터의 신호를 누적하여 새로운 벡터로 합성하는 정보 누적 연산
- `att`: 두 벡터의 신호를 일정 비율로 반영하여 두 벡터 사이 벡터로 보간하는 정보 선택 연산
- `mean`: 두 벡터의 신호를 균등 비율로 반영하여 두 벡터 사이 벡터로 보간하는 정보 선택 연산
- `prod`: 두 벡터의 신호를 요소별로 결합하여 공통으로 활성화된 성분을 강조하는 정보 여과(상호작용) 연산
- `cat`: 후속 레이어(e.g. mlp, linear, etc.)에서 정보 누적과 선택, 여과를 수행하기 위하여 두 벡터의 정보를 보존함

## 표기

### idx

- $u=0,1,2,\cdots,M-1$: target user
- $i=0,1,2,\cdots,N-1$: target item
- $j \in R_{u}^{+} \setminus \{i\}$: history items of target user (target item $i$ is excluded)
- $M,N$ is padding idx

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
- $y_{u,i}, \hat{y}_{u,i}$: $(u,i)$ pair interaction probability

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

### 사용자 표현

- user global behavior:

$$
p_{u}=\mathrm{embedding}(u)
$$

- user conditional preference:

$$
c_{u}=\mathrm{bam}(p_{u}, \forall \psi_{j}, \forall \psi_{j})
$$

- user refined representation:

$$
z_{u}=\mathrm{comb}(p_{u}, c_{u})
$$

### 아이템 표현

- item global behavior:

$$
q_{i}=\mathrm{embedding}(i)
$$

- item conditional preference:

$$
c_{i}=\mathrm{bam}(q_{i}, \forall \phi_{v}, \forall \phi_{v})
$$

- item refined representation:

$$
z_{i}=\mathrm{comb}(q_{i}, c_{i})
$$

### 매칭 함수

- concatenation(agg) & mlp(matching):

$$
z_{u,i}=\mathrm{mlp}_{\mathrm{ReLU}}(z_{u} \oplus z_{i})
$$

- logit:

$$
x_{u,i}=h^{T}(W \cdot z_{u,i}+b)
$$

- prediction:

$$
\hat{y}_{u,i}=\sigma(x_{u,i})
$$

### 목적 함수

$$
\mathcal{L}_{\mathrm{ELBO}}:= \sum_{(u,i)\in\Omega}{\left(\mathrm{NLL} + \sum_{j \in R_{u}^{+} \setminus \{i\}}{\mathrm{KL}^{(u,j)}} + \sum_{j \in R_{u}^{+} \setminus \{i\}}{\mathrm{KL}^{(i,j)}} \right)}
$$

- apply `bce` to pointwise `nll`:

$$
\mathcal{L}_{\mathrm{BCE}}:=-\sum_{(u,i)\in\Omega}{y_{u,i}\ln{\hat{y}_{u,i}} + (1-y_{u,i})\ln{(1-\hat{y}_{u,i})}}
$$

- apply `bpr` to pairwise `nll` (only log-likelihood):

$$
\mathcal{L}_{\mathrm{BPR}}:=-\sum_{(u,pos,neg)\in\Omega}{\ln{\sigma(x_{u,pos} - x_{u,neg})}}
$$

## 데이터 셋

- movielens latest small (2018) [`link`](https://grouplens.org/datasets/movielens/latest/)
    - interaction density is relatively high
    - the reliability of individual observations is relatively low

- last.fm 2k (2011) [`link`](https://grouplens.org/datasets/hetrec-2011/)
    - interaction density is relatively low
    - the reliability of individual observations is relatively high

- amazon luxury beauty small 5-core (2018) [`link`](https://cseweb.ucsd.edu/~jmcauley/datasets/amazon_v2/)
    - interaction density is relatively low
    - the reliability of individual observations is relatively low
    - user history length is relatively short

- amazon digital music small 5-core (2014) [`link`](https://cseweb.ucsd.edu/~jmcauley/datasets/amazon/links.html)
    - interaction density is relatively low
    - the reliability of individual observations is relatively low
    - user history length is relatively long

## 실험

데이터 셋을 `trn`, `val`, `tst` 각각 8:1:1 로 사용자 기준 계층적 분할하였다. 추가로 `leave-one-out` 데이터 셋을 구성하여 학습 조기종료 시점을 모니터링하는데 사용하였다. `opt`(`trn`, `val`) 데이터 셋에 대하여 1:4, `msr`(`tst`, `loo`) 데이터 셋에 대하여 1:99 비율로 네거티브 샘플링을 적용하였다. 초기 10 epoch 에 대해서는 조기종료 여부를 모니터링하지 않았고, 11 epoch 부터 모니터링하여 `ndcg@10` 이 최대 5회 개선되지 않을 경우 학습을 조기종료하였다.

네 데이터 셋 중 밀집도가 비교적 높은 movielens latest small (2018) 에서는 일부 사용자의 상호작용 이력이 2,000 건이 넘는데 반해, 상위 10% 사용자의 이력은 400 건이다. 효율성을 도모하기 위하여 T선별 점수 기준 상위 400 건의 이력만을 활용하였다. 선별 점수로는 상호작용 빈도와 TF-IDF 를 활용하였으며, TF-IDF 를 활용하였을 때 추가적인 성능 개선이 있었다. TF-IDF 를 활용할 때는 문서를 상호작용 이력에, 단어(혹은 토큰)를 해당 이력의 구성자에 대응하여 점수를 산출하였다.

논문에서는 어텐션 스코어 함수로서 NAIS(He et al., 2018)에서 제안한 함수들을 적용하였으나, 본 레파지토리에서는 내적을 적용하였다. 또한 본 레파지토리에서는 어텐션 스코어의 사전 분포와 변분 분포로서 로그 정규 분포를 활용하여 실험을 진행하였다. 이때 사전 분포의 표준편차는 1.0, 변분 분포의 표준편차는 0.1 으로 임의 고정하였다. 다만, 해당 수치는 최적값이라 볼 순 없으며, 데이터 신뢰도가 낮거나 희소성이 높을수록 변분 분포의 표준편차가 클 때 성능이 개선되는 양상을 보였다.

- movielens latest small (2018) [`notebook`](/notebooks/movielens/)

- last.fm 2k (2011) [`notebook`](/notebooks/lastfm/)

- amazon luxury beauty small 5-core (2018) [`notebook`](/notebooks/abeauty/)

- amazon digital music small 5-core (2014) [`notebook`](/notebooks/amusic/)