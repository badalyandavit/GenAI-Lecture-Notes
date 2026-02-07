## 11.1 Random utility view: utility plus noise

A unifying model is:

$$  
u_i = s_i + \varepsilon_i  
$$

- ($s_i$) is the deterministic score produced by the model.
    
- ($\varepsilon_i$) is random noise capturing unobserved factors.
    

A ranking is produced by sorting latent utilities:  
$$  
i \succ j \quad \Longleftrightarrow \quad u_i > u_j  
$$

Different assumptions on the noise distribution ($\varepsilon_i$) yield different probabilistic ranking models and losses.

---

## 11.2 Bradley–Terry from logistic noise (pairwise comparisons)

### Pairwise preference probability

If the noise induces a logistic distribution on utility differences, we obtain the Bradley–Terry form:  
$$  
P(i \succ j)=\sigma(s_i-s_j)=\frac{\exp(s_i)}{\exp(s_i)+\exp(s_j)}  
$$

### Pairwise loss

For observed preference (i \succ j):  
$$  
\mathcal{L}_{\text{BT}}=\log\left(1+\exp\left(-(s_i-s_j)\right)\right)  
$$

Key properties:

- Only score differences matter.
    
- Adding a constant to all scores in a group does not change probabilities.
    
- Strongly aligned with preference datasets.
    

---

## 11.3 Thurstone–Mosteller from Gaussian noise (probit comparisons)

Assume Gaussian noise:  
$$  
\varepsilon_i \sim \mathcal{N}(0,\sigma^2)  
$$

Then the difference ($u_i-u_j$) is Gaussian with mean ($s_i-s_j$) and variance ($2\sigma^2$), giving:  
$$  
P(i \succ j)=\Phi\left(\frac{s_i-s_j}{\sqrt{2}\sigma}\right)  
$$  
where ($\Phi$) is the standard normal CDF.

The corresponding loss is the probit negative log-likelihood:  
$$  
\mathcal{L}_{\text{probit}}=-\log \Phi\left(\frac{s_i-s_j}{\sqrt{2}\sigma}\right)  
$$

Notes:

- Similar qualitative behavior to Bradley–Terry.
    
- Less common in large neural ranking systems mainly due to numerical and computational convenience (sigmoid is cheaper than normal CDF).
    

---

## 11.4 Plackett–Luce from i.i.d. Gumbel noise (full permutations)

A classic result: if ($\varepsilon_i$) are i.i.d. Gumbel (Type I extreme value), then the probability that an item has the maximum utility follows softmax. Extending this sequentially yields Plackett–Luce for whole permutations.

Permutation probability:  
$$  
P(\pi)=\prod_{k=1}^{m}\frac{\exp(s_{\pi_k})}{\sum_{j=k}^{m}\exp(s_{\pi_j})}  
$$

Negative log-likelihood:  
$$  
\mathcal{L}_{\text{PL}}(\pi)=  
-\sum_{k=1}^{m}\left(  
s_{\pi_k}-\log\sum_{j=k}^{m}\exp(s_{\pi_j})  
\right)  
$$

### Relationship to Bradley–Terry

If there are only two items ((m=2)), PL reduces exactly to BT:  
$$  
P(\pi_1= i)=\frac{\exp(s_i)}{\exp(s_i)+\exp(s_j)}  
$$

So you can view:

- BT as a 2-item choice model
    
- PL as a sequential choice model that defines a distribution over full rankings
    

---

## 11.5 Multinomial logit (MNL) as one-step choice

For a single choice among items:  
$$  
P(i)=\frac{\exp(s_i)}{\sum_j \exp(s_j)}  
$$

This is the multinomial logit model. PL can be viewed as repeating this choice among the remaining items at each rank position.

This matters because softmax classifiers and language models use exactly this choice rule.

---

## 11.6 Mallows model (distance-based ranking)

Mallows assigns probability to permutations based on distance from a reference permutation ($\pi_0$):  
$$  
P(\pi)\propto \exp(-\theta d(\pi,\pi_0))  
$$

- (d(\cdot,\cdot)) is a permutation distance (often Kendall tau distance).
    
- (\theta) controls concentration around (\pi_0).
    

This is conceptually clean but less common at scale because:

- Computing with permutation distances can be expensive.
    
- Training is harder than PL-style objectives.
    

---

## 11.7 Preference learning in language models

Language-model alignment datasets often contain comparisons:  
$$  
(x, y^+, y^-)  
$$  
meaning (y^+) is preferred to (y^-) for prompt (x).

### 11.7.1 Pairwise preference loss (BT style over candidates)

Define a scoring function over full responses (often a reward model):  
$$  
s^+ = f_\theta(x, y^+),\qquad s^- = f_\theta(x, y^-)  
$$

BT-style loss:  
$$  
\mathcal{L}_{\text{pref}}=\log\left(1+\exp\left(-(s^+-s^-)\right)\right)  
$$

This learns a utility function up to an additive constant per prompt.

---

## 11.8 Direct Preference Optimization (DPO)

DPO trains the policy (the language model) directly from preference pairs without an explicit reward model, using a reference policy ($\pi_{\text{ref}}$) to control drift.

### 11.8.1 Standard DPO objective

Let ($\pi_\theta$) be the policy being trained. DPO defines a logit for the preference:  
$$  
\Delta(x)=  
\left[  
\log \pi_\theta(y^+|x)-\log \pi_\theta(y^-|x)  
\right]

\left[  
\log \pi_{\text{ref}}(y^+|x)-\log \pi_{\text{ref}}(y^-|x)  
\right]  
$$

Then the loss is:  
$$  
\mathcal{L}_{\text{DPO}}=  
-\log \sigma\left(\beta \Delta(x)\right)  
$$

- (\beta>0) is a temperature-like scaling parameter.
    
- The reference term keeps optimization stable and acts like a regularizer.
    

### 11.8.2 How to read DPO as a ranking model

DPO is essentially a Bradley–Terry style pairwise model, but the “scores” are **log-likelihood differences** of the policy relative to a reference.

That is:

- Items are full responses (y)
    
- The score is a likelihood-based utility
    
- Preferences are enforced pairwise via a logistic link
    

---

## 11.9 Relationship between objectives in LM systems

### Token-level likelihood training (MLE)

At each timestep:  
$$  
p_\theta(v_t=v\mid x,y_{<t})=\frac{\exp(s_v)}{\sum_{u\in\mathcal{V}}\exp(s_u)}  
$$  
MLE loss:  
$$  
\mathcal{L}_{\text{MLE}}=-\sum_t \log p_\theta(y_t\mid x,y_{<t})  
$$

This is one-step MNL choice repeated across timesteps.

### Sequence-level preference training

Preference objectives operate over whole candidates (y) rather than tokens:

- Pairwise reward modeling: learns ($f_\theta(x,y)$)
    
- DPO: directly updates ($\pi_\theta$) using log-likelihood ratios
    

A useful mental model:

|Objective|Items being ranked|Score type|Structure|
|---|---|---|---|
|MLE|tokens (v)|logits (s_v)|listwise choice per step|
|Pairwise preference|sequences (y)|learned utility (f_\theta(x,y))|pairwise|
|DPO|sequences (y)|log-likelihood ratio vs reference|pairwise|

---

## 11.10 Implementation notes (what usually breaks)

### Stable computation

Use log-sum-exp for listwise losses:  
$$  
\log\sum_j \exp(s_j)  
$$  
should be computed stably (shift by max logit).

### Query-grouped losses

Ranking losses must be computed within each query group:

- PL needs all items in the group
    
- Pairwise needs pairs drawn from the same group
    

### Pair sampling

For pairwise training, the sampling strategy matters:

- Random pairs are cheap but weak
    
- Hard pairs are informative but can be noisy
    
- A common compromise is semi-hard negatives (high score but not top)
    

---

## 11.11 Summary

- Random utility models unify ranking: ($u_i=s_i+\varepsilon_i$).
    
- BT corresponds to logistic-linked pairwise preferences.
    
- Thurstone–Mosteller corresponds to probit-linked pairwise preferences.
    
- PL corresponds to sequential softmax choice and defines a full distribution over permutations.
    
- Language models use softmax choice at token level and preference objectives at sequence level.
    
- DPO can be seen as BT-style ranking over candidates using likelihood ratios relative to a reference.
    

---

## References

- Christiano, P. F., Leike, J., Brown, T., Martic, M., Legg, S., & Amodei, D. (2017). Deep reinforcement learning from human preferences. In _Proceedings of NeurIPS 2017_.
    
- Luce, R. D. (1959). _Individual Choice Behavior: A Theoretical Analysis_. Wiley.
    
- Mallows, C. L. (1957). Non-null ranking models. _Biometrika, 44_(1/2), 114–130.
    
- McFadden, D. (1974). Conditional logit analysis of qualitative choice behavior. In P. Zarembka (Ed.), _Frontiers in Econometrics_. Academic Press.
    
- Plackett, R. L. (1975). The analysis of permutations. _Journal of the Royal Statistical Society: Series C (Applied Statistics), 24_(2), 193–202.
    
- Rafailov, R., Sharma, A., Mitchell, E., Ermon, S., Manning, C. D., & Finn, C. (2023). Direct preference optimization: Your language model is secretly a reward model. In _Proceedings of NeurIPS 2023_.
    
- Thurstone, L. L. (1927). A law of comparative judgment. _Psychological Review, 34_(4), 273–286.
    
- Train, K. E. (2009). _Discrete Choice Methods with Simulation_ (2nd ed.). Cambridge University Press.
    
- Bradley, R. A., & Terry, M. E. (1952). Rank analysis of incomplete block designs: I. The method of paired comparisons. _Biometrika, 39_(3/4), 324–345.
    
- Ouyang, L., Wu, J., Jiang, X., Almeida, D., Wainwright, C., Mishkin, P., et al. (2022). Training language models to follow instructions with human feedback. _arXiv:2203.02155_.
