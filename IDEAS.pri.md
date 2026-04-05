**Top recommendation: Start with the self-improvement loop (GEA + EGGROLL) as the #1 highest-leverage, lowest-cost/highest-benefit move.**  
All four proposals are extremely similar (they’re basically four phrasings of the same 2025–2026 paper synergies). They agree on the core non-additive levers: GEA/EGGROLL as the engine, CALM/Miras for compression & memory, qTTT for test-time adaptation, and CIB for structural compression. The differences are mostly framing and order, not substance.

The **strongest / most valuable** ideas share three traits:
- They create **compounding positive feedback** (better evolution → better models → better data → even better evolution).
- They are **backward-compatible** with your existing Tiny Shakespeare proxy and src/ structure.
- They have **tiny implementation cost** (many are <50 lines) but unlock 3–20× gains in convergence speed, quality, or throughput.

### Prioritized Roadmap (Pursue in This Order)

**Phase 0 (Next 1–3 days – Quickest Wins, Highest ROI)**  
These are literally “code today” changes with massive leverage. Do them first.

1. **GEA multi-island + diversity bonus + experience migration** (from Proposal 1)  
   - Add population-scale islands (10–20 parallel runs, different seeds + tiny architectural variants).  
   - Every 5–10 generations: migrate top 20 % (weights + successful CIB patterns + CALM K schedules).  
   - Fitness 2.0: quality × compression × generalization (add 3–4 synthetic reasoning tasks) + pairwise cosine diversity bonus.  
   - **Cost**: <50 lines in `src/evolution/`.  
   - **Benefit**: 3–5× faster convergence + emergent capabilities (exactly what GEA was built for). This turns GEA from “auxiliary optimizer” into the primary training signal.  
   - **Why strongest**: Every other improvement (CALM, Miras, qTTT, CIB) gets amplified automatically by a better evolutionary engine.

2. **Switch fitness function to include generalization term + meta-evolution** (Proposals 1 & 3)  
   - Let EGGROLL also evolve its own hyperparameters (population size, mutation schedule, migration rate).  
   - **Cost**: trivial (already in your EGGROLL wrapper).  
   - **Benefit**: self-tuning optimizer → open-ended discovery of better workflows.

**Phase 1 (Next 1–2 weeks – Highest Leverage Single Change)**  
3. **Adaptive / learned K in CALM** (mentioned as quick win in all four proposals)  
   - Add a tiny router (or just a small MLP head) that predicts chunk size K per context. Train it with EGGROLL (no backprop needed).  
   - Small K for reasoning steps, large K for high-throughput generation.  
   - **Cost**: very low (`src/model/` + one line in CALM encoder).  
   - **Benefit**: 8–15× effective throughput + better quality (directly from CALM paper scaling behavior). This is the single highest-ROI line change you can make right now.

**Phase 2 (Weeks 2–4 – Compound Everything)**  
4. **CIB as regularizer inside GEA evolution + task-conditioned heads** (Proposals 1, 3, 4)  
   - Penalize verbose chains during evolution; add separate bottleneck heads for math/code/common-sense.  
   - **Cost**: medium (extend CIB loss in training loop).  
   - **Benefit**: 3–5× fewer reasoning tokens with zero accuracy drop. Makes real-time CoT possible on edge.

5. **Miras v2 + recurrent cache + qTTT hierarchical** (Proposals 2 & 4)  
   - Stack 4–8 Miras blocks with lightweight recurrence + cache of top-N vectors from GEA.  
   - Hierarchical qTTT (coarse then fine).  
   - **Cost**: medium (model/ folder).  
   - **Benefit**: long-context jumps (5 % → 40–60 % on LongBench-style) + deeper memory without KV cache explosion.

**Lower priority (do after the above)**  
- Full scaling to 1B–7B params + OpenWebText/synthetic data (all proposals). Valuable but higher cost; the evolutionary loop will generate the data for you once Phase 0–1 are running.  
- Full meta-evolution of evolution rules or latent-thoughts heads (nice-to-haves; they shine once the core loop is strong).

### Why This Order Beats the Other Proposals’ Suggested Orders
- Proposal 1 correctly puts the self-improvement loop first — that’s the rocket fuel.  
- Proposals 2–4 start with “scale params/data” or “accuracy-per-parameter” first. Those are important but **much higher cost** and deliver less immediate compounding. The evolutionary engine makes scaling trivial later.  
- All quick wins listed across the proposals (diversity bonus, learned K, fitness generalization term) are in my Phase 0/1. They are the real lowest-cost/highest-benefit moves.

**Bottom line**: The architecture is already correct. The massive gains (5–20×+ overall) come from **making the GEA/EGGROLL loop the primary driver** instead of an auxiliary tool, then layering the CALM adaptive-K change on top. Everything else compounds from there.

If you want, I can:
- Write the exact <50-line diffs for the GEA multi-island + diversity bonus right now.
- Or the adaptive-K router pseudocode + where it slots into your CALM encoder.
- Or update your `docs/experiments.md` with this prioritized plan + ablation list.

Just say which one to start coding first.

