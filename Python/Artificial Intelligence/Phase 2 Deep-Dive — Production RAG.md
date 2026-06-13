# Phase 2 Deep-Dive — Production RAG over a Real Corpus
## A week-by-week build, designed to become your strongest interview artifact

*Companion to the main timeline. This is the phase that gets you hired, so it gets its own document with repo structure, a concrete dataset, target metrics, and the exact design decisions you must defend with numbers. Everything here assumes you finished Phase 1 — and it reuses what you built there rather than rebuilding it.*

---

## What you carry forward from Phase 1 (don't rebuild these)

You did not finish Phase 1 with throwaway code. Three of its artifacts become load-bearing tools in Phase 2 — recognizing this reuse is itself a senior signal (you build evaluation infrastructure once and apply it everywhere):

1. **The eval harness.** In Phase 1 it scored a classification prompt on accuracy, cost, and p95 latency. In Phase 2 you *extend* the same harness with retrieval metrics (precision@k, recall@k, MRR) and end-to-end metrics (faithfulness, answer relevance). Same runner, same Postgres-backed result storage, new scorers. If you find yourself writing a second eval system, stop — you're duplicating.
2. **The prompt registry.** Your RAG answer-synthesis step *is* a prompt, and it belongs in the same versioned registry. When you tune the "answer using only this context, cite sources" prompt, you version it exactly like Phase 1 and diff quality across versions.
3. **The CI eval gate.** The GitHub Actions workflow that failed a build on accuracy regression now also fails on *retrieval* regression. A chunking change that quietly drops recall@5 should turn the build red.

The mental continuity matters: Phase 1 taught you to *measure an unreliable component*. Phase 2 is the same discipline applied to a pipeline with five places to measure instead of one. RAG without the Phase 1 measurement habit is exactly the naive RAG that doesn't get people hired.

**A note on why RAG is the employment centerpiece.** Almost every company deploying LLMs on their own data is doing RAG. The supply of engineers who can wire a tutorial-grade "embed and grab top-5" pipeline is enormous; the supply who can *measure retrieval quality, defend chunking with numbers, and explain why hybrid beat pure-vector on their corpus* is small. This phase is engineered to land you in the second group.

---

## The dataset (pick one, commit fully)

Theory dies without a real corpus. Choose **one** and use it the entire phase — switching datasets midway wastes your labeling effort.

- **Recommended: a public technical documentation corpus.** Something like the Python standard-library docs, the AWS service docs, or a large open-source project's docs (Kubernetes, PostgreSQL). Reasons it's ideal: it's large enough to be non-trivial, it has real structure (headings, code blocks, cross-references) that makes chunking *matter*, and it contains exact identifiers (function names, error codes) that expose the dense-vs-sparse tradeoff vividly — pure semantic search will famously fumble an exact API name, which gives you a clean story for why you added hybrid search.
- **Alternative: a public Q&A or knowledge dataset** with built-in questions and answers (e.g. a subset of a published QA dataset). The advantage is you get *labeled* question→answer pairs for free, which jump-starts your golden set.
- **Avoid:** tiny toy corpora (a handful of PDFs). They make every retrieval strategy look identical and give you nothing to measure or defend.

Whatever you pick, the single most important early task is building a **labeled query set** — roughly 50–100 questions, each tagged with which chunk(s) genuinely answer it. This is the ground truth your entire eval rests on. It is tedious. Do it anyway; it is the difference between "I think hybrid is better" and "hybrid lifted recall@5 from 0.61 to 0.84 on my labeled set."

---

## New concepts this phase adds (beyond the main doc's vocabulary)

The main timeline defined embedding, vector space, ANN, HNSW/IVFFlat, chunking, hybrid search, BM25, RRF, cross-encoder, precision@k/recall@k/MRR, faithfulness, and the failure modes. Assume those. Here are the *additional* concepts you need at implementation depth:

**Embedding model selection & dimensionality.** Not all embedding models are equal. They differ in vector dimension (more dimensions = richer representation but more storage and slower search), context length (how much text one embedding can represent), and domain fit (a model trained on general web text may underperform on dense technical docs). The senior move: treat embedding-model choice as an *evaluated* decision, not a default — swap two models and measure retrieval quality on your labeled set. The **MTEB** (Massive Text Embedding Benchmark) leaderboard is the standard reference for comparing them, but your corpus is the only benchmark that actually counts.

**Index build vs query tradeoffs in `pgvector`.** HNSW and IVFFlat each have build-time parameters that trade index-build speed, query speed, and recall. For HNSW: `m` (graph connectivity) and `ef_construction` (build-time search breadth) set quality at build time; `ef_search` tunes the recall/latency tradeoff at query time. You don't need to memorize optimal values — you need to *know these knobs exist* and be able to say "I tuned `ef_search` to hit my p95 latency target while holding recall above 0.8." That sentence is senior; "I used pgvector defaults" is not.

**Metadata filtering.** Real RAG rarely searches the whole corpus blindly. You attach metadata to chunks (source document, section, date, access level) and filter on it — "search only the current API version's docs," "only documents this user may see." The interaction between metadata filtering and ANN indexing has real performance subtleties (pre-filter vs post-filter), and knowing they exist signals production experience.

**Query transformation.** The user's raw question is often a poor search query. Techniques: **query rewriting** (have an LLM rephrase the question into a better retrieval query), **multi-query** (generate several phrasings and union the results), and **HyDE** (Hypothetical Document Embeddings — generate a hypothetical *answer* and embed *that* to search with, since an answer is often closer in vector space to the real answer than the question is). These are measurable upgrades; add them only if your eval shows they help, and you'll have a great "I tried HyDE, it helped on X but not Y" story.

**Contextual retrieval / context injection.** A chunk ripped from its document loses context ("it" referring to something three chunks back). A modern technique prepends a short LLM-generated summary of the chunk's place in the document before embedding it, measurably improving retrieval. Worth knowing as a current best practice you can name.

**Faithfulness vs answer-relevance vs context-precision (the eval triad).** Going deeper than the main doc: *faithfulness* asks "is every claim in the answer supported by the retrieved context?" (catches hallucination). *Answer relevance* asks "does the answer address the question?" (catches off-topic but grounded answers). *Context precision/recall* asks "did retrieval surface the right chunks in the right order?" (isolates whether a bad answer is retrieval's fault or generation's fault). The senior insight is that these three *decompose* failure — when an answer is bad, this triad tells you *which stage* to fix. `ragas` computes all three.

**The retrieval-caps-generation principle.** Burn this in: **no prompt can fix bad retrieval.** If the right chunk never makes it into the context, the model cannot answer faithfully — it can only hallucinate or refuse. This is why you evaluate retrieval *independently* and first. Most naive RAG debugging wastes days tuning the answer prompt when the real fault is upstream in retrieval.

---

## Week-by-week build

### Week 4 — Ingestion, chunking, and your first measurable baseline

**Goal:** a working but deliberately *naive* end-to-end RAG pipeline, plus the labeled query set — so that everything you do in Weeks 5–6 can be measured *against this baseline*. You cannot prove improvement without a baseline; building the naive version first is intentional, not a detour.

**Build:**
- An **ingestion pipeline**: load the corpus, clean it, split into chunks. Start with simple recursive chunking (split on paragraphs/sections down to a token limit, with small overlap).
- An **embedding step**: embed each chunk with one chosen model, store vectors + text + metadata in `pgvector`.
- **Naive retrieval**: embed the query, cosine top-k, stuff chunks into a synthesis prompt (versioned in your Phase 1 registry), return an answer.
- **The labeled query set**: 50–100 questions, each annotated with the chunk(s) that answer it. This is your week's most valuable output even though it's the least glamorous.
- **Extend the Phase 1 eval harness** with precision@k and recall@k scorers that run against the labeled set.

**By end of week you can say:** "My naive baseline gets recall@5 of [X] and faithfulness of [Y]" — a concrete number every later improvement is measured against.

### Week 5 — Retrieval quality: hybrid search and re-ranking

**Goal:** beat the baseline measurably by fixing *retrieval*, the stage that caps everything downstream.

**Build, measuring after each change:**
- **Sparse retrieval** via Postgres full-text search (BM25-style), running alongside the dense `pgvector` search.
- **Fusion** of the two ranked lists with RRF. Re-run the eval — you should see recall climb, especially on queries containing exact identifiers. *This is your headline result.* Record the before/after numbers.
- **Cross-encoder re-ranking**: take the fused top-20, re-score with a cross-encoder, keep the top-5. Re-run the eval — precision@k should rise. Note the latency cost this adds (re-ranking isn't free) so you can speak to the quality/latency tradeoff.
- **A chunking experiment**: try one alternative chunking strategy (e.g. structure-aware vs your recursive baseline) and measure which wins. Now you can *defend* your chunking choice with data instead of taste.

**By end of week you can say:** "Hybrid + re-rank lifted recall@5 from [baseline] to [new] and precision@5 from [X] to [Y], at a p95 latency cost of [Z]ms — here's the table." That sentence is the core of your interview story.

### Week 6 — Generation quality, robustness, and the deployable service

**Goal:** turn a good retriever into a trustworthy, deployable RAG service, and finish the eval picture.

**Build:**
- **Citation-grounded synthesis**: the answer prompt must produce inline citations pointing at the exact source chunk, and you verify citations actually support their claims (this is where *faithfulness* and *citation drift* get real).
- **The full eval triad** via `ragas`: faithfulness, answer relevance, context precision — so you can decompose any failure to its stage.
- **One query-transformation experiment** (query rewriting or HyDE), kept only if the eval justifies it. Either outcome is a good story.
- **Redis caching** of embeddings and frequent query results (reuse your Phase 0 caching instincts), with measured latency impact.
- **Robustness handling**: what happens when retrieval finds nothing relevant? The system should *say so*, not hallucinate. Test the empty-context and adversarial-question cases explicitly.
- **The FastAPI `/query` endpoint**: streams cited answers, returns the source chunks alongside, and records cost + latency per request (reuse Phase 0 instrumentation).
- **CI gate**: wire the retrieval and faithfulness evals into the Phase 1 GitHub Actions workflow so a regression fails the build.

**By end of week you have:** a deployable, instrumented, eval-gated RAG service with citations — the artifact you screen-share.

---

## Suggested repo structure

A clean repo is itself an interview signal — it shows you build maintainable systems, not scripts. Something like:

```
rag-service/
├── README.md                  # the most important file — see below
├── docker-compose.yml         # postgres+pgvector, redis, app — one command to run
├── pyproject.toml
├── .github/workflows/ci.yml   # lint, tests, AND the eval gate
├── src/
│   ├── ingestion/             # loading, cleaning, chunking
│   │   ├── chunkers.py        # recursive + structure-aware, swappable
│   │   └── pipeline.py
│   ├── embedding/             # embedding model wrapper (swappable, like Phase 0's provider swap)
│   ├── retrieval/
│   │   ├── dense.py           # pgvector search
│   │   ├── sparse.py          # postgres FTS
│   │   ├── fusion.py          # RRF
│   │   └── rerank.py          # cross-encoder
│   ├── synthesis/             # answer generation + citation logic
│   ├── api/                   # FastAPI app, /query endpoint
│   └── eval/                  # EXTENDS phase-1 harness, doesn't replace it
│       ├── retrieval_metrics.py   # precision@k, recall@k, MRR
│       ├── ragas_runner.py        # faithfulness, relevance, context precision
│       └── golden_set/            # your labeled query set lives here, version-controlled
├── tests/
└── notebooks/                 # the experiments: chunking A/B, embedding model comparison, HyDE trial
```

Two things interviewers notice: the swappable components (chunkers, embedding models, retrievers) showing you design for experimentation, and the `eval/` directory existing *at all* — most candidates' RAG repos have no evaluation code, which tells the interviewer everything.

---

## The README is the interview (write it deliberately)

Your README is what a hiring engineer reads first, and often all they read before deciding to dig in. Structure it to front-load the senior signals:

1. **One-paragraph what-and-why.**
2. **Architecture diagram** (the five-stage pipeline — you already have the visual vocabulary for this).
3. **Results table** — the headline. Baseline vs hybrid vs hybrid+rerank, across recall@5, precision@5, faithfulness, and p95 latency. Numbers, not adjectives. This table is the single most persuasive thing in the repo.
4. **Key design decisions, each with its justification** — "Recursive chunking (beat structure-aware by X on recall, see notebook)," "pgvector over Pinecone because [reasons, and when I'd switch]," "added re-ranking despite +Zms latency because precision rose Y."
5. **What I'd do at 10× / 100× scale** — shows you think past the demo.
6. **Known limitations** — naming them is a confidence signal, not a weakness.

The decisions-with-justification section is what separates this from a tutorial repo. A tutorial repo says *what* it did; yours says *why*, with evidence.

---

## Target metrics (what "good" looks like)

Exact numbers depend on your corpus, so these are directional targets to aim for and, more importantly, to *measure honestly* even when you miss them — an honest "recall@5 is 0.78 and here's why it's not higher" beats a suspicious 0.99.

- **Recall@5**: aim for a clear, measured lift over your naive baseline — the *delta* matters more than the absolute.
- **Precision@5**: should rise noticeably after re-ranking.
- **Faithfulness** (ragas): high — this is the anti-hallucination metric, and low faithfulness is disqualifying for a RAG system.
- **p95 query latency**: set a target you can defend (e.g. "under 2s end-to-end including re-rank") and tune `ef_search` / re-rank depth to hit it.
- **Cost per query**: tracked, because you built that habit in Phase 0.

The framing that wins: not "my system is good" but "here is the baseline, here is each improvement, here is what it cost, and here is the tradeoff I chose." That is what senior judgment *looks like* on paper.

---

## The design decisions you must defend with numbers

Interviewers probe decisions. For each of these, you should have a number and a sentence ready — this list is effectively your prep sheet:

- **Why this chunking strategy?** → "It beat [alternative] on recall@5 by [X]; see the A/B notebook."
- **Why this embedding model?** → "I compared two; this one scored higher on my labeled retrieval set, not just on MTEB."
- **Why hybrid over pure vector?** → "Pure vector missed exact-identifier queries; hybrid lifted recall from [X] to [Y]."
- **Why re-ranking, given the latency cost?** → "Precision@5 rose from [X] to [Y]; the +[Z]ms was within my p95 budget."
- **Why pgvector and not a dedicated vector DB?** → "One system, transactional consistency with metadata, scales past where people assume; I'd move to [X] above [N] vectors or [Q] QPS."
- **How do you know it's not hallucinating?** → "Faithfulness eval at [score], plus citation verification, plus an explicit empty-context refusal path."
- **How would this fail in production?** → name lost-in-the-middle, stale index, citation drift, and your mitigation for each.

If you can answer these six with numbers, you can hold a senior RAG conversation at any of your target companies. If you can only answer them with adjectives, you've built a tutorial. The whole phase is engineered to put numbers behind every one.

---

## How this sets up Phase 3

The RAG service you just built becomes a *tool* in Phase 3 — your bounded agent will call this `/query` endpoint as its `search` capability. So build the endpoint clean and well-documented; future-you (and your agent) depends on it. The continuity is the point: each phase's artifact is the next phase's building block, which is exactly the story you tell an interviewer about how you think — in systems, measured at every seam, reused rather than rebuilt.

---

*Do the labeled query set first and do it honestly — every number in your interview story traces back to it. A RAG repo with a real eval harness and a results table puts you ahead of the large majority of candidates, who have neither.*
