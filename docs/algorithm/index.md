# Algorithm Overview & Modes

> [!TIP]
> For visual diagrams, animated video explanations, and an interactive 2D simulator, see
> the **[Visual Architecture & Options Guide](visual_guide.md)**.

The GRIC clustering algorithm is designed for high-speed, sequential distance-based clustering. It
processes incoming data frames one by one, grouping them into clusters defined by **anchor frames**
under a maximum distance constraint (`rlim`).

## High-Level Workflow

For every new frame `fi` in a sequence, the algorithm performs the following overall steps:

```mermaid
flowchart TD
    Start([New Frame fi]) --> Step1[1. Normalize Probabilities]
    Step1 --> Step2[2. Calculate Mixed Probabilities]
    Step2 --> Step3[3. Rank Candidate Clusters]
    
    Step3 --> Step4{Target Selection Mode?}
    
    Step4 -->|Greedy Mode| Greedy[Greedy Priority<br/>mixed_probs * gprobs]
    Step4 -->|Entropy Mode| Entropy[Entropy Minimization<br/>Shannon Info Gain]
    
    Greedy --> Check[Compute Distance dfc]
    Entropy --> Check
    
    Check --> Match{dfc < rlim?}
    
    Match -->|Yes| Assign[Assign fi to Cluster]
    Match -->|No| Prune[Prune Candidates via Geometry]
    
    Prune --> Remaining{Any active candidates?}
    Remaining -->|Yes| Step4
    Remaining -->|No| Create[5. Create New Cluster]
    
    Assign --> End([Frame Processed])
    Create --> End
```

### The 5 Core Steps

1.  **Normalize Probabilities**: Normalize the prior assignment probabilities `prob(cj)` of existing
    clusters so they sum to 1.0.
2.  **Calculate Mixed Probabilities**: If temporal prediction is active, mix priors with
    Markov transitions (`-tm`) or fuzzy trajectory similarity (`-predf`). When binary sequence
    prediction (`-pred`) is active, the top predicted cluster IDs are tested first.
3.  **Rank Candidate Clusters**: Sort the existing clusters to determine the initial search order.
    If geometrical probability (`-gprob`) is active, candidates are dynamically tracked using the
    normalized running posterior distribution `entropy_p_current`.
4.  **Select & Check Candidates**: Iterate through candidates using either **Greedy** or
    **Entropy** target selection:
    - Compute the distance `dfc(fi, cj)` to the candidate anchor.
    - If `dfc < rlim`, assign the frame to this cluster and update transition/recency probabilities.
    - If `dfc > rlim`, use geometric pruning to eliminate other impossible candidate clusters.
5.  **Create New Cluster**: If all existing candidate clusters are checked or pruned,
    establish a new cluster with `fi` as its anchor.

---

## Target Selection Modes: Greedy vs. Entropy

A key performance factor in GRIC is deciding *which* cluster to measure next during Step 4. GRIC
offers two primary modes for target selection:

### 1. Greedy Mode (Default)
In Greedy Mode, the algorithm selects targets solely by prioritizing the most likely match first:
- **Standard Greedy (no `-gprob`)**: It sequentially checks candidates in a static order sorted by
  prior probability.
- **Dynamic Greedy (with `-gprob`)**: It dynamically updates the candidates' geometrical probability
  after each distance measurement and selects the active candidate with the highest current
  posterior probability (`entropy_p_current`).

**Advantage**: Very low computation overhead per step. It works exceptionally well when there is a
high-probability candidate (e.g., in continuous streams with high temporal correlation).

---

### 2. Entropy Mode (`-entropy`)
In Entropy Mode, instead of greedily testing the most likely candidate, the algorithm treats target
selection as an information-theory optimization problem:

- **Adaptive Entropy Gating (`-entropy_gate`, `-entropy_first_gate`)**: If posterior Shannon
  entropy is already lower than the gating threshold (e.g., confident distribution), the engine
  measures the top candidate directly, saving evaluation overhead.
- **Dominant Leader Shortcut (`-entropy_leader`)**: When active, if a candidate's probability
  meets or exceeds `-entropy_leader_cutoff` (default: 0.50), it is measured immediately.
- Otherwise, it evaluates a subset of active candidates (up to `-entropy_max_targets`, default 15)
  and calculates the **Expected Shannon Entropy** across all active candidate hypotheses:
  
  \[
  \mathbb{E}[H(T)] = \sum_{c_j} P(c_j) \cdot H(T \mid c_j \text{ is true})
  \]

- **Hypothesis Evaluation**: For each hypothetical true cluster $c_j$, the precomputed
  `consistency_mask` identifies which candidate clusters $k$ survive triangle inequality pruning
  ($|d(k, T) - d(c_j, T)| \le 2 r_{\text{lim}}$).
- **Posterior Entropy**: The normalized probabilities of surviving candidates define the
  conditional posterior distribution, computing hypothetical Shannon entropy $H(T \mid c_j)$.
- The algorithm selects the candidate target $T$ that **minimizes** $\mathbb{E}[H(T)]$ (maximizing
  expected information gain per distance calculation).

**Advantage**: Significantly reduces the number of expensive distance computations in
high-dimensional or noisy datasets where simple greedy paths struggle.

---

### Synergy and Complementarity

`gprob` (Geometrical Probability) and `-entropy` (Shannon Entropy Optimization) are highly
complementary because they handle different roles in the target selection process:

- **`gprob` acts as a probability generator**: It calculates the spatial probability distribution
  based on geometric similarity and historical co-measurements. It answers the question:
  *Where is the frame likely located in space?*
- **`-entropy` acts as a decision-theoretic scheduler**: It utilizes the probability distribution
  provided by `gprob` to calculate the expected Shannon information gain. It answers the question:
  *Which cluster anchor should I measure next to minimize search ambiguity as fast as possible?*

When both are enabled, `gprob` dynamically updates and refines the candidates' likelihoods after
each distance calculation, and `-entropy` uses this fresh distribution to schedule the next optimal
anchor to query. This synergistic relationship reduces the number of expensive metric evaluations
to a theoretical minimum.

---

## Detailed Components

Explore the technical details of the sub-components:
*   **[Geometric Pruning Mechanisms](pruning.md)**: How the triangle inequality, 4-point, and
    5-point pruning eliminate candidates.
*   **[Probability & Prediction Models](prediction.md)**: Details on transition matrices,
    temporal sequence forecasting, and geometric match probabilities.
