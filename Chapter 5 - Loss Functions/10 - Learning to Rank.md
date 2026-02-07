This is my favorite part, hence the two-part notes.
## 10.1 Ranking tasks and notation

**Ranking**: given a context (often called a query) and a set of candidate items, produce an ordering that matches relevance.

Examples:

- Information retrieval: rank documents for a query.
    
- Recommender systems: rank items for a user.
    
- Reranking: rank candidate outputs from a generator model.
    
- Preference learning: rank options using pairwise preferences.
    

### Basic setup

- Context (query, user state, prompt): (x)
    
- Items: (${i_1,\dots,i_m}$)
    
- Scoring function:  
    $$  
    s_i = f_\theta(x, i)  
    $$
    
- Predicted ranking (a permutation):  
    $$  
    \hat{\pi} = \operatorname{argsort}_i; s_i \quad \text{(descending)}  
    $$
    

The key object is the **ordering** induced by scores, not the absolute value of the scores.

---

## 10.2 Supervision types in ranking

Ranking datasets typically come in one of three forms:

### A) Pointwise labels

Each item gets an independent label:

- Binary relevance: ($y_i \in {0,1}$)
    
- Graded relevance: ($y_i \in {0,1,2,3,\dots}$)
    
- Real-valued utility: ($y_i \in \mathbb{R}$)
    

### B) Pairwise preferences

Comparisons of the form:

- ($i \succ j$) meaning item (i) should be ranked above (j)
    
- Often encoded as ($y_{ij}\in{0,1}$)
    

### C) Listwise supervision

A whole ranked list or permutation is provided:

- A permutation (\pi) for items in a query group
    
- Or graded labels for all items in the list, which implies a target ordering
    

---

## 10.3 Evaluation metrics (what ranking cares about)

Ranking losses are usually surrogates. Metrics define what “good ranking” means.

### Discounted cumulative gain (DCG) and NDCG

For a ranked list, let ($\text{rel}_r$) be the relevance of the item at rank position (r).

DCG@(k):  
$$  
\mathrm{DCG@}k=\sum_{r=1}^{k}\frac{2^{\mathrm{rel}_r}-1}{\log_2(r+1)}  
$$

Ideal DCG@(k) (IDCG@(k)) is DCG@(k) under the best possible ordering.

Normalized DCG@(k):  
$$  
\mathrm{NDCG@}k = \frac{\mathrm{DCG@}k}{\mathrm{IDCG@}k}  
$$

NDCG is popular for graded relevance because it is:

- Top-heavy (early ranks matter more)
    
- Scale-sensitive (handles relevance levels)
    

### Mean reciprocal rank (MRR)

For a single query, let ($\mathrm{rank}^*$) be the rank position of the first relevant item.

$$  
\mathrm{RR}=\frac{1}{\mathrm{rank}^*},\qquad  
\mathrm{MRR}=\frac{1}{Q}\sum_{q=1}^{Q}\mathrm{RR}_q  
$$

MRR is common when there is a single “correct” answer or a small relevant set.

### Precision@k and Recall@k (binary relevance)

$$
\text{Precision@}k = \frac{\#\text{ relevant in top }k}{k}
$$
  
$$
\mathrm{Recall@}k =
\frac{\#\text{ relevant in top }k}{\#\text{ relevant total}}
$$


### Mean average precision (MAP) (binary relevance)

Average precision (AP) for a query is:  
$$  
\mathrm{AP}=\frac{1}{R}\sum_{r=1}^{m}\mathrm{Precision@}r \cdot \mathbf{1}[\text{item at }r\text{ is relevant}]  
$$  
where (R) is the total number of relevant items for that query.

Then:  
$$  
\mathrm{MAP}=\frac{1}{Q}\sum_{q=1}^{Q}\mathrm{AP}_q  
$$

---

## 10.4 Three training paradigms

### 10.4.1 Pointwise ranking

Treat each item independently and predict a label or score.

Score model:  
$$  
s_i=f_\theta(x,i)  
$$

Examples of pointwise losses:

- Regression (real-valued or graded labels):  
    $$  
    \mathcal{L}_{\text{MSE}} = \sum_i (s_i - y_i)^2  
    $$
    
- Classification (graded labels as classes):  
    $$  
    \mathcal{L}_{\text{CE}} = -\sum_i \log p_\theta(y_i\mid x,i)  
    $$
    

**Pros**

- Simple and scalable
    
- Works when labels are reliable and well-calibrated
    

**Cons**

- Does not optimize ordering directly
    
- Sensitive to label scaling and calibration
    
- Can be misaligned with NDCG and other permutation metrics
    

---

### 10.4.2 Pairwise ranking (Bradley–Terry and logistic pairwise loss)

Train on pairs ((i,j)) with preference signal (i \succ j).

#### Bradley–Terry preference probability

Define:  
$$  
P(i \succ j) = \frac{\exp(s_i)}{\exp(s_i)+\exp(s_j)}=\sigma(s_i-s_j)  
$$  
where (\sigma(z)=\frac{1}{1+e^{-z}}).

Negative log-likelihood for one preferred pair:  
$$  
\mathcal{L}_{\text{pair}}=\log\left(1+\exp\left(-(s_i-s_j)\right)\right)  
$$

This is the standard RankNet-style logistic pairwise loss.

**Pros**

- Directly uses relative comparisons
    
- Invariant to adding a constant to all scores
    
- Matches human preference data naturally
    

**Cons**

- Optimizes local comparisons, not full permutations
    
- Requires sampling pairs (can be expensive)
    
- Pairwise data can be inconsistent (cycles)
    

---

### 10.4.3 Listwise ranking (Plackett–Luce and permutation likelihood)

Listwise methods model the probability of an entire ranked list.

Let (\pi) be a permutation of ({1,\dots,m}). Under the Plackett–Luce (PL) model:  
$$  
P(\pi)=\prod_{k=1}^{m}\frac{\exp(s_{\pi_k})}{\sum_{j=k}^{m}\exp(s_{\pi_j})}  
$$

Negative log-likelihood (ListMLE style):  
$$  
\mathcal{L}_{\text{list}}(\pi)=  
-\sum_{k=1}^{m}\left(  
s_{\pi_k}-\log\sum_{j=k}^{m}\exp(s_{\pi_j})  
\right)  
$$

**Intuition**

- At each rank position (k), choose the next item among remaining items with probability proportional to (\exp(s)).
    
- This explicitly models competition among all remaining candidates.
    

**Pros**

- Closest to the structure of ranking metrics (permutation-based)
    
- Uses full list context (global competition)
    
- Often performs well when you can train with full query groups
    

**Cons**

- More expensive (needs lists per query)
    
- Sensitive to how you define the target permutation when labels are tied
    
- Requires stable log-sum-exp implementations
    

---

## 10.5 Practical considerations (the parts that matter in real systems)

### Query grouping and batching

Ranking is almost always **grouped by query** (or user). Losses should be computed within each group.

- Pointwise can ignore groups, but evaluation is group-based.
    
- Pairwise and listwise require query groups directly.
    

### Negative sampling

In many settings you only observe positives. You must sample negatives.

Common strategies:

- Random negatives
    
- Hard negatives (top-scoring non-relevant items)
    
- In-batch negatives (use other items in the batch)
    

Hard negatives usually speed up learning but can increase label noise.

### Position bias and click feedback

If training labels come from clicks, labels are biased by exposure (items higher in the list are more likely to be clicked). Common fixes include:

- Randomization interleaving
    
- Inverse propensity scoring (IPS) reweighting
    
- Click models that separate relevance from examination probability

---

## 10.6 Connection to language models

Learning-to-rank shows up in language-model pipelines in two main ways.

### A) Reranking candidate outputs

Given a prompt (x), a generator proposes candidates ({y_1,\dots,y_m}). A reranker assigns scores:  
$$  
s_k = f_\theta(x, y_k)  
$$  
and returns (\arg\max_k s_k) or the top (k) candidates.

This is used in:

- Best-of-(n) sampling with a reward model
    
- Tool selection and routing
    
- Preference-based filtering
    

### B) Token selection as a choice model

At decoding step (t), a language model produces logits (${s_v}_{v\in\mathcal{V}}$) over the vocabulary (\mathcal{V}):  
$$  
p_\theta(v_t=v\mid x, y_{<t})=\frac{\exp(s_v)}{\sum_{u\in\mathcal{V}}\exp(s_u)}  
$$

This is a **multinomial logit (softmax) choice rule** over tokens. Autoregressive generation applies this choice repeatedly:  
$$  
p_\theta(y\mid x)=\prod_t p_\theta(y_t\mid x, y_{<t})  
$$

A useful perspective:

- Tokens are competing items at each step.
    
- Softmax is a listwise choice over the vocabulary at that step.
    
- Generation is sequential choice under competition.
    

This view becomes especially important when we move from likelihood training to preference training (Chapter 11).

---

## 10.7 Summary

- **Pointwise** learns absolute scores or labels.
    
- **Pairwise** learns relative order via comparisons.
    
- **Listwise** learns a permutation likelihood and directly models global competition.
    
- Ranking metrics are permutation-focused, especially top-heavy ones like NDCG.
    
- Language-model systems frequently perform ranking through reranking, preference learning, and token-level softmax choice.
    

---

## References

- Bradley, R. A., & Terry, M. E. (1952). Rank analysis of incomplete block designs: I. The method of paired comparisons. _Biometrika, 39_(3/4), 324–345.
    
- Burges, C., Shaked, T., Renshaw, E., Lazier, A., Deeds, M., Hamilton, N., & Hullender, G. (2005). Learning to rank using gradient descent. In _Proceedings of ICML 2005_.
    
- Cao, Z., Qin, T., Liu, T.-Y., Tsai, M.-F., & Li, H. (2007). Learning to rank: From pairwise approach to listwise approach. In _Proceedings of ICML 2007_.
    
- Järvelin, K., & Kekäläinen, J. (2002). Cumulated gain-based evaluation of IR techniques. _ACM Transactions on Information Systems, 20_(4), 422–446.
    
- Liu, T.-Y. (2009). _Learning to Rank for Information Retrieval_. Foundations and Trends in Information Retrieval.
    
- Luce, R. D. (1959). _Individual Choice Behavior: A Theoretical Analysis_. Wiley.
    
- Plackett, R. L. (1975). The analysis of permutations. _Journal of the Royal Statistical Society: Series C (Applied Statistics), 24_(2), 193–202.
    
- Xia, F., Liu, T.-Y., Wang, J., Zhang, W., & Li, H. (2008). Listwise approach to learning to rank: Theory and algorithm. In _Proceedings of ICML 2008_.
    

