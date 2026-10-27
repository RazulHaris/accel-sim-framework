# R2: a separate prefetch-queue injection path (spec)

**Status: not implemented.** Saved 2026-10-27; implementation waits for approval of Step 2 of the SK match (results/SK_match_qv100.md). Recorded as SPEC OQ-38.

R2 is our reading. SnakeCharmer (Tang et al., MICRO 2026, arXiv 2610.30990v1, §III-A) states only: "We empirically configure all candidate prefetch requests to share a single 128-entry prefetch queue." It does not say how queued prefetches reach the memory system.

## Behaviour
- **Option `snake_pq_path`:**
  - 0 = current: prefetches enter the shared L1 miss queue.
  - 1 = R2.
  - Default 0, unchanged.
- **In R2,** issued prefetches wait in the 128-entry prefetch queue and go directly to the L1 → interconnect injection point. They never enter the L1 miss queue, which then carries demand only. No miss-queue reserve is needed.
- **Injection limit:** at most ONE request per cycle per SM in total, shared between the miss queue and the prefetch queue (same as baseline). A separate-port variant may exist only as a labelled sensitivity option, never the default.
- **`snake_pq_arb`:**
  - 0 = demand first: a prefetch is sent only when the miss queue is empty.
  - 1 = round-robin when both queues have requests.
- **`snake_pq_mshr`:**
  - 0 = allocate the MSHR when the prefetch is sent. Re-check the tags and the MSHR then; drop the prefetch if its line is already present or requested.
  - 1 = allocate at enqueue. A demand to the same line merges into it and counts as late.
- **Throttling** still applies.
- **Duplicate checks** at enqueue stay as today.

## Stats
- prefetch-queue occupancy, wait time, drops (queue full)
- arbitration wins per queue
- demand injection delay vs baseline
- demand MISS_QUEUE_FULL vs baseline

## Verification
- The off-regression must stay byte-identical with `snake_pq_path 0`.

## Step 2 runs (when approved)
- R2 × `snake_pq_arb` {0, 1} × `snake_pq_mshr` {0, 1}, on top of the better Step 1 config.
- Same workloads as Step 1, base QV100, 100M.

## Record
- An OQ citing SnakeCharmer §III-A (SPEC OQ-38).
