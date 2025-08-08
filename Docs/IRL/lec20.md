# $$Inverse\ Reinforcement\ Learning$$

## Decision making

1. Deterministic case

$$a_1,...,a_T=\arg\max_{a_1,...,a_T}\sum_{t=1}^Tr(s_t,a_t)$$

$$\text{With: }s_{t+1}=f(s_t,a_t),a_t=u(s_t)$$

2. Stochastic case

$$\pi =\arg\max_{\pi}\mathbb{E}_{s_{t+1}\sim p(s_{t+1}|s_t,a_t),a_t\sim \pi(a_t|s_t)}[r(s_t,a_t)]$$

$$\text{With: } s_{t+1}\sim p(s_{t+1}|s_t,a_t),a_t\sim \pi(a_t|s_t)$$

## Human perspective

1. Imitation learning
    - Copy the **actions** performed by the expert.
    - No **reasoning** about the outcomes of actions.

2. Human Imitation learning
    - Copy the **intent** of the expert. (understand the underlying task)
    - Might take very **different** actions. (many optimal solutions exist)

    
