# Findings: Training vs Evaluation Limitations

## Finding 1 --- DPO Training Limitation

### Observation

MS MARCO training qrels provide only shallow positive supervision. In
our current preprocessing (Cell 7), we further simplify the data by
keeping only the first positive passage for each query.

### Why this matters

DPO learns from **preferences** between generated pseudo-documents. The
reward is based on whether retrieval successfully ranks the selected
positive passage highly.

If a query actually has multiple relevant passages, but only one is
kept:

-   A generated pseudo-document that retrieves the retained passage
    receives a high reward.
-   A generated pseudo-document that retrieves other relevant passages
    may receive a much lower reward or no reward.

As a result, the DPO reward becomes a **noisy proxy** for true retrieval
quality.

### Consequence

The model may learn to prefer generations that optimize for one retained
passage instead of generations that retrieve a broader set of relevant
passages.

### Important note

The larger limitation comes from **MS MARCO itself**, whose training
judgments are shallow. Keeping only the first positive passage is an
additional simplification that may discard useful supervision.

------------------------------------------------------------------------

## Finding 2 --- nDCG Evaluation Limitation (Pooling Problem)

### Observation

Evaluation is performed on **TREC DL 2019**, not on the MS MARCO
training qrels.

Unlike MS MARCO, DL19 provides graded relevance labels (0--3), allowing
nDCG to reward highly relevant documents more than partially relevant
ones.

### Limitation

TREC DL judgments are created using a **pooling** process.

Only documents that were examined by human assessors receive relevance
labels.

If our retrieval system discovers a genuinely relevant document that was
**never judged**, `trec_eval` treats it as having relevance 0.

### Consequence

A retrieval method may actually improve search quality by finding new
relevant passages, yet receive little or no credit from nDCG because
those passages were never included in the judgment pool.

### Key distinction

-   **Finding 1 (DPO)** affects **training** by providing an imperfect
    reward signal.
-   **Finding 2 (nDCG)** affects **evaluation** by potentially
    underestimating retrieval quality due to incomplete judgments.
