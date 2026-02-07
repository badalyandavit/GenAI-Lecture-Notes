
|Target type|Support|Typical likelihood|Loss (NLL) implies|Point estimate you get “for free”|
|---|--:|---|---|---|
|Real-valued regression|$\mathbb{R}$|Normal|MSE (if variance fixed)|Mean|
|Robust regression|$\mathbb{R}$|Laplace, Student-t, Cauchy|MAE or heavy-tail robust loss|Median (Laplace)|
|Quantile prediction|$\mathbb{R}$|Asymmetric Laplace (view)|Pinball (quantile) loss|Chosen quantile $\tau$|
|Class probability|${1,\dots,C}$|Categorical|Cross-entropy|MAP class|
|Imbalanced classification|${1,\dots,C}$|Modified CE|Focal loss|MAP class|
|Ranking a list|permutations|Plackett–Luce|Listwise ranking NLL|Ordering|
|Bounded proportion|$(0,1)$|Beta|Beta NLL|Mean of Beta|
|Counts|${0,1,2,\dots}$|Poisson, NegBin|Count NLL|Mean rate|
|Positive durations|$(0,\infty)$|Gamma, lognormal, Weibull|Duration NLL|Mean or median|
|Angles (circular)|$[-\pi,\pi)$|von Mises|Circular NLL|Circular mean|
|Multimodal regression|$\mathbb{R}^d$|Mixture (MDN)|Mixture NLL|Mixture mean or modes|
