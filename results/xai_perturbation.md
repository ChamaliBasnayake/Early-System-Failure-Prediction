# XAI perturbation check — sem_final_N5_s11, N=5
Integrated Gradients, 50 steps, baseline = empty window (PAD embeddings, mean scalars). Completeness gap: mean 0.1388, max |gap| 2.7004 (logit units).
Windows: 5 true positives and 0 hard-negative false positives above the 1 % FAR threshold (0.4097) on validation; plus the 50 highest-scoring positives and 50 highest-scoring negatives regardless of threshold (non-empty windows).

Deleting the top-k positively attributed items vs deleting k random items (mean score drop; frac_top_bigger = share of windows where the attributed deletion hurt more):

|                                                 |   n |   mean_score |   drop_top |   drop_random |   frac_top_bigger |
|:------------------------------------------------|----:|-------------:|-----------:|--------------:|------------------:|
| ('top-50 scoring negatives (any score)', 1)     |  50 |       0.6825 |     0.3148 |        0.0497 |              0.8  |
| ('top-50 scoring negatives (any score)', 3)     |  50 |       0.6825 |     0.443  |        0.0439 |              0.8  |
| ('top-50 scoring negatives (any score)', 5)     |  50 |       0.6825 |     0.511  |        0.1543 |              0.8  |
| ('top-50 scoring positives (any score)', 1)     |  50 |       0.2092 |     0.0929 |        0.0218 |              0.6  |
| ('top-50 scoring positives (any score)', 3)     |  50 |       0.2092 |     0.1217 |        0.0216 |              0.76 |
| ('top-50 scoring positives (any score)', 5)     |  50 |       0.2092 |     0.1299 |        0.041  |              0.72 |
| ('true positives (above 1 % FAR threshold)', 1) |   5 |       0.6775 |     0.4197 |       -0.0038 |              1    |
| ('true positives (above 1 % FAR threshold)', 3) |   5 |       0.6775 |     0.4721 |        0.0259 |              1    |
| ('true positives (above 1 % FAR threshold)', 5) |   5 |       0.6775 |     0.5024 |       -0.0133 |              1    |

**Reading guide:** if `drop_top` ≫ `drop_random` and `frac_top_bigger` is well above 0.5, the highlighted items genuinely drive the score. A small gap means the model's decision is spread over many items (or is driven by the scalar context).