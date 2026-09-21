# Gen-AI Pipeline for Duplicate Detection

A multimodal proof of concept that finds and resolves duplicate product records by combining Claude vision, local text embeddings, LLM judgment and human review.

## Overview

Enterprise product catalogs accumulate duplicate records over time, degrading search, pricing consistency, recommendations and inventory accuracy. Exact and fuzzy string matching breaks down when:

- the same product is described in different words,
- different products have near-identical names, or
- the difference is visual and only visible in the photograph.

This pipeline works in semantic space instead of character space, and keeps a human data steward in the loop for uncertain decisions.

## How it works

1. **Ingest.** Load the Amazon Berkeley Objects (ABO) catalog (264,904 products), scope it to one category (`CELLULAR_PHONE_CASE`, 88,588 products) and sample it with a fixed random seed. A visual check on one image runs before any API spend.
2. **Fingerprint.** Claude (vision) writes a short description of each product image. It is combined with the product type and domain and embedded locally with `all-MiniLM-L6-v2` (384 dimensions), so the embedding step runs on your machine.
3. **Find candidates.** Compute a cosine similarity matrix and keep pairs at or above a threshold (0.85).
4. **Judge.** Claude compares each candidate pair and returns JSON: `are_duplicates`, `proposed_instance_id`, `merged_product_type`, `confidence` and `reasoning`. The decision is then routed:

   | Status | When |
   | --- | --- |
   | `AUTO_MERGE` | confidence of 0.90 or higher and the pair is a duplicate (the proposed golden record must be one of the two products) |
   | `AUTO_KEEP_SEPARATE` | confidence of 0.90 or higher and the pair is not a duplicate |
   | `PENDING_REVIEW` | confidence below 0.90, or a malformed response, for a human steward |

5. **Record lineage.** Every decision is stored in SQLite (`data_lineage.db`: `lineage_sources`, `lineage_events`) with the two products compared, the outcome, confidence, reasoning, similarity score, pipeline version and a `run_id`. The audit query and summary are scoped to the current run.

The AI provider is created in a single initialization block, so it can be swapped for another hosted or on-premise model.

## Tech stack

| Area | Tools |
| --- | --- |
| Language | Python, Jupyter Notebook |
| LLM and vision | Anthropic Claude (`claude-sonnet-4-6`) |
| Embeddings | sentence-transformers (`all-MiniLM-L6-v2`), scikit-learn cosine similarity |
| Data | pandas, Kaggle ABO dataset via `kagglehub`, SQLite |

## Results

The notebook prints a summary after each run: products analyzed, candidate pairs flagged, and counts of auto-merge, auto-keep-separate, pending review and failed decisions. Numbers vary slightly between runs because the image descriptions come from an LLM.

In the saved run (10 phone cases, seed 42), 13 of the 45 possible pairs passed the 0.85 similarity threshold and none failed:

| Outcome | Pairs | Similarity | Claude's confidence |
| --- | --- | --- | --- |
| `AUTO_KEEP_SEPARATE` | 10 | 0.854 to 0.897 | 0.92 to 0.95 |
| `PENDING_REVIEW` | 3 | 0.925 to 0.960 | 0.62 to 0.82 |
| `AUTO_MERGE` | 0 | none | none |

The three pairs sent to human review were the three most similar ones, so the pipeline escalates the pairs that look most alike instead of deciding them on its own. Similarity proposes candidates and the LLM decides. This sample contained no confirmed duplicates, so no merge was made; the merge path is covered by the routing logic but has not been exercised on real duplicates.

## Run it

1. Create a `.env` file next to the notebook:

   ```text
   ANTHROPIC_API_KEY=your-key-here
   ```

2. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

3. Open and run the notebook top to bottom:

   ```bash
   jupyter notebook DQ_flag_duplicate_Final_clean.ipynb
   ```

The dataset is downloaded automatically by `kagglehub` on first run.

## Known limitations

- This is a proof of concept on a sample of 10 products. There is no labeled ground truth yet, so no precision or recall is reported.
- Pairwise similarity is quadratic in the number of products. Production use needs blocking and an approximate nearest-neighbor index.
- The fingerprint uses only product type, domain and the generated image description, which inflates similarity within a single category: 13 of 45 pairs passed 0.85 in the saved run although none was confirmed as a duplicate.

## Next steps

- Build a labeled set of duplicate pairs and hard negatives, then report precision and recall for the candidate and judgment stages separately.
- Add blocking and an ANN index (FAISS or HNSW) for scale.

## Skills demonstrated

Generative AI, multimodal embeddings, prompt engineering with structured output, human-in-the-loop design, data quality, data lineage and governance.
