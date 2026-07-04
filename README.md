\# Adaptive Intelligent Recommendation \& Decision System



A recommendation engine that segments users by behavior, predicts preference

likelihood, generates personalized recommendations via matrix factorization,

and adapts over time using reinforcement learning.



\## Problem Statement



The MovieLens 100K dataset (943 users, 1682 items, 100,000 ratings) is 93.7%

sparse — the vast majority of user-item pairs have no rating. This sparsity

is why naive approaches like global averages fail, and why a multi-layered

approach (segmentation, prediction, factorization, adaptation) is needed.



\## Approach



1\. \*\*User Segmentation (Unsupervised Learning)\*\* — K-Means clustering (k=5,

&#x20;  selected via elbow method) grouped users into behavioral segments: critical/

&#x20;  inconsistent raters, generous consistent fans, steady moderate raters, high-

&#x20;  volume power users, and typical users.



2\. \*\*Preference Prediction (Supervised Learning)\*\* — Logistic Regression and a

&#x20;  Neural Network (32→16 hidden units) were both trained to predict whether a

&#x20;  user would rate an item 4+. Both achieved \~71.8% accuracy, well above the

&#x20;  \~55% naive majority-class baseline. Logistic Regression was selected for

&#x20;  production due to comparable performance with greater interpretability.



3\. \*\*Recommendation Generation (Matrix Factorization)\*\* — Truncated SVD (20

&#x20;  components) generates personalized Top-N recommendations. An initial

&#x20;  implementation had a critical bug: filling unrated entries with 0 caused

&#x20;  the model to learn toward zero across the board, producing an RMSE of

&#x20;  2.64 — worse than a naive mean predictor (1.12). Fixing this with per-user

&#x20;  mean-centering before factorization (and adding the mean back after

&#x20;  reconstruction) brought RMSE down to 0.99, a 12% improvement over baseline.



4\. \*\*Cold Start Handling\*\* — New users with no rating history receive

&#x20;  popularity-weighted recommendations from users who share at least one

&#x20;  known preference, filtered to items with 5+ ratings for reliability.



5\. \*\*Adaptive Refinement (Reinforcement Learning)\*\* — An epsilon-greedy

&#x20;  multi-armed bandit (epsilon=0.1) demonstrates how the system could adapt

&#x20;  recommendations from real click/ignore feedback rather than relying on a

&#x20;  static model. In simulation, it correctly identified the best-performing

&#x20;  item among 10 candidates and selected it in 82% of rounds after 1000

&#x20;  rounds of feedback.



\## Results



| Metric | Value |

|---|---|

| Dataset sparsity | 93.70% |

| Logistic Regression accuracy | 71.78% |

| Neural Network accuracy | 71.79% |

| Naive baseline RMSE | 1.1239 |

| Mean-centered SVD RMSE | 0.9888 |

| Precision@10 | 0.2850 |

| Bandit best-arm selection rate | 82% (1000 rounds) |



\## Tech Stack



Python, NumPy, Pandas, Scikit-learn, Matplotlib, Seaborn, Streamlit



\## Project Structure

