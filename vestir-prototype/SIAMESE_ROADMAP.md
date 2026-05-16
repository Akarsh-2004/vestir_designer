# Siamese Retrieval Roadmap

## Objective
Add a Siamese reranker on top of the existing embedding retrieval stack without replacing current production matching.

## Baseline (Current)
- Candidate generation: `embedding-sidecar` vectors + cosine ranking from server storage.
- Match serving: `/api/extension/match` in `server/index.mjs`.

## Proposed Hybrid
1. Keep current vector retrieval for top-K candidate recall.
2. Add Siamese reranker for top-K refinement with pairwise compatibility scoring.
3. Blend score:
   - `final_score = 0.65 * cosine_score + 0.35 * siamese_score`
4. Expose score components for explainability in fit UI.

## Model Design
- Twin encoder:
  - Image branch initialized from SigLIP/CLIP encoder checkpoint.
  - Optional metadata branch for category, season, and formality.
- Loss options:
  - Primary: triplet loss (hard-negative mining).
  - Secondary: contrastive BCE for clicked/not-clicked interaction pairs.
- Output:
  - 256D normalized embedding for pair distance.
  - Optional compatibility head (`0..1`) for rank fusion.

## Data Strategy
- Positive pairs:
  - User-saved outfits.
  - High dwell/click + add-to-fit behavior.
- Hard negatives:
  - Same category but rejected/low dwell.
  - Color-clashing candidates within same anchor context.
- Sampling:
  - Category-balanced batches.
  - Seasonal and formality-balanced mini-batches to reduce shortcut learning.

## Evaluation
- Offline:
  - Recall@10, Recall@20
  - NDCG@10
  - Pairwise AUC for compatibility head
- Online:
  - Outfit save rate uplift
  - Click-through on recommended cards
  - Time-to-first-save reduction

## API Evolution Plan
- New optional rerank endpoint:
  - `POST /api/items/rerank-siamese`
  - Input: anchor item + candidate IDs + contextual intent
  - Output: reranked list with `siamese_score` and `final_score`
- Integrate behind feature flag:
  - `SIAMESE_RERANK_ENABLED=1`

## Rollout
1. Stage 1: Offline training + eval only.
2. Stage 2: Shadow mode scoring in production logs.
3. Stage 3: 10% traffic A/B with reranker enabled.
4. Stage 4: Full rollout when win criteria are met.
