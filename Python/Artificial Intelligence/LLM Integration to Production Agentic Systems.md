# From LLM Integration to Production Agentic Systems
## A teaching timeline — built to be defended in interviews

*Written the way a senior engineer with 20+ years would mentor a sharp junior: every term defined the first time it appears, every concept given both the theory and the production reality, and every phase ending in something you can point at and explain under pressure.*

---
file:///Users/dev/Downloads/Visual%20companion.html#train
---

## How to read this document

You already have the engineering maturity (6 years production, Rails/Python/React/AWS) and the classical ML foundation (Ng's courses, ongoing ML work). So I'm not going to insult you with "what is a function." What I *am* going to do is the thing that's hard to get from tutorials: explain **why** each technique exists, **what breaks** when you don't understand it, and **how a senior talks about it** in a room full of skeptical interviewers.

Each phase has five parts:
- **The mental model** — the intuition a senior carries in their head.
- **Concepts & vocabulary** — the "fancy words," each defined plainly, then deepened.
- **The project** — what you build, and why this specific thing.
- **Stack & rationale** — not just *what* tools, but *why these*.
- **Defense checklist** — the exact claims you should be able to make and back with numbers.

A rule I want you to internalize before we start: **in GenAI, the seniority signal is not "I can make it work" — it's "I can measure it, bound it, and tell you when not to build it."** Everything below is organized around that.

**Pacing:** ~10–15 focused hours/week over 14 weeks. Theory is interleaved with code on purpose — you said you learn by building, and concepts stick when you've felt them break.

**Default stack (recurring):** Python 3.12 · FastAPI · PostgreSQL (+ `pgvector`) · Redis · AWS · Pytest · Docker · GitHub Actions · Anthropic + OpenAI SDKs.

---

## A foundational vocabulary (read this first)

Before Phase 0, here are the load-bearing terms the whole field rests on. We'll go deeper later, but you need these in muscle memory.

**LLM (Large Language Model).** A neural network — almost always a *transformer* — trained on enormous text corpora to predict the next *token* given previous tokens. That's it, mechanically: a very good next-token predictor. Everything impressive it does (reasoning, coding, translation) is an emergent consequence of doing that prediction extremely well at scale. Knowing this deflates a lot of mystique and helps you reason about failure: when an LLM "lies," it isn't deceiving you — it's predicting plausible-looking tokens with no ground-truth check.

**Token.** The atomic unit an LLM reads and writes. Not a word and not a character — a sub-word fragment produced by a *tokenizer*. "Unbelievable" might be three tokens (`un`, `believ`, `able`). Why you care: you are billed per token, latency scales with tokens *generated*, and context limits are measured in tokens. Token-thinking is cost-thinking.

**Transformer.** The neural network architecture (introduced in the 2017 paper "Attention Is All You Need") underneath nearly every modern LLM. Its key innovation is the **attention mechanism** — a way for the model, when processing one token, to weigh the relevance of every other token in the input. That's why an LLM can connect "it" in a sentence back to the noun it refers to twenty words earlier. You don't need to implement one, but you must be able to say this sentence in an interview without flinching.

**Context window.** The maximum number of tokens (input + output) a model can consider at once. Think of it as the model's working memory — it has *no* memory beyond what's in the window. Every "memory" feature you'll build (conversation history, RAG) is really just a strategy for deciding what to *put into* this finite window. Misunderstanding this is the root cause of most naive-implementation failures.

**Inference.** Running a trained model to get an output (as opposed to *training*, which produces the model). When you "call the LLM," you're running inference. In your world, inference is an API call with a latency cost and a dollar cost — you almost never train.

**Prompt.** The input text you give the model. Sounds trivial; it's an entire engineering discipline (Phase 1) because the prompt is the *only* control surface you have over a model you can't retrain.

**Hallucination.** When a model produces fluent, confident, *wrong* output. Critical insight: this is not a bug to be patched — it's intrinsic to next-token prediction. The model optimizes for plausibility, not truth. Your job as an engineer is to *constrain* hallucination (with retrieval, with citations, with validation), never to assume it's "fixed."

**Grounding.** Tying a model's output to verifiable source data so it can't freely invent. RAG (Phase 2) is the dominant grounding technique. "Grounded" answers can cite where they came from.

**Agent.** An LLM placed inside a loop where it can take *actions* (call tools, query databases, hit APIs), observe the results, and decide what to do next — rather than just emitting one block of text. The whole back half of this document is about building these reliably.

Keep this glossary open as you work. By Phase 3 these words should feel like `git` does to you now.

---

## Phase 0 — Foundations & instrumentation (Week 1)

### The mental model
You already know how to call an LLM API. This week is about calling it *like the systems engineer you are*: typed, cached, observable, and cost-accounted from the first line. Juniors build a working demo and then bolt on monitoring after it's on fire. Seniors instrument first. The reason is selfish — every later phase generates more LLM calls, and if you can't *see* token spend and latency now, you'll be blind exactly when it matters.

### Concepts & vocabulary

**Tokenization & token economics.** A *tokenizer* splits text into tokens; different model families use different tokenizers, so the same sentence costs different amounts on different models. **Input tokens** (your prompt) and **output tokens** (the model's reply) are usually priced differently, with output often several times more expensive. Practical consequence: a chatty system prompt is a fixed tax on every single request, while verbose outputs are the variable cost that explodes under load. Senior move: you can look at a feature and estimate its per-request cost before building it.

**Sampling parameters.** When generating each token, the model produces a probability distribution over all possible next tokens. *How* it picks from that distribution is controlled by:
- **Temperature** — a knob (typically 0–1, sometimes higher) controlling randomness. `0` = always pick the most likely token (deterministic, repeatable). Higher = more varied, more "creative," less predictable. For evals and tool-calling you want low temperature (you need reproducibility); for brainstorming you want it higher.
- **Top-p (nucleus sampling)** — instead of considering all tokens, only consider the smallest set whose probabilities sum to *p* (e.g. 0.9), then sample from those. It's an alternative way to control randomness by truncating the long tail of unlikely tokens.

Knowing these lets you answer "why is my output non-deterministic?" — a question that trips up people who treat the API as a black box.

**Structured output.** Getting the model to return machine-parseable data (usually JSON) instead of prose. Two mechanisms: **JSON mode** (the API constrains output to valid JSON) and **tool/function calling** (you give the model a schema and it returns arguments matching it). Why it matters: "please respond in JSON" in a prompt works in a demo and fails ~5% of the time in production, and 5% is a catastrophe at scale. Schemas enforced by the API are the difference.

**Streaming (Server-Sent Events / SSE).** Instead of waiting for the full response, you receive tokens as they're generated. This is a UX necessity (users tolerate a streaming answer far better than a 10-second blank screen) and a systems concern (streaming interacts awkwardly with load balancers, timeouts, and buffering — you'll feel this in Phase 5).

**Idempotency, retries, and backoff.** LLM APIs fail — *rate limits* (HTTP 429, you're sending too fast), timeouts, transient 5xx errors. The seniors' pattern is **exponential backoff with jitter**: on failure, wait a bit and retry, doubling the wait each time, with a random offset ("jitter") so that a fleet of your servers doesn't retry in synchronized waves and hammer the API simultaneously (the "thundering herd" problem). You know this from distributed systems; it applies cleanly here.

### The project — "LLM Gateway"
A FastAPI service fronting both Anthropic and OpenAI behind one clean interface. Requirements that make it senior-grade:
- **Pydantic models** (Pydantic is Python's data-validation library — it turns untyped dicts into validated, typed objects) for every request and response, so malformed data fails loudly at the boundary.
- **Per-request token + cost accounting** written to Postgres — every call records model, input tokens, output tokens, and computed USD cost.
- **Redis response cache** keyed on a hash of `(model, parameters, prompt)`. Identical requests return instantly and for free. (Redis is an in-memory data store — microsecond reads — ideal for caching and ephemeral state.)
- **Structured logging** with a trace ID per request so you can follow one request across logs.
- **Exponential backoff with jitter** on 429/5xx.
- One **streaming** `/chat` endpoint.

### Stack & rationale
FastAPI because it's async-native (LLM calls are I/O-bound — your server should handle other requests while waiting on the model) and gives you typed request validation for free. Postgres for durable cost records you can query. Redis for the cache because cache reads must be faster than the thing they replace. This is the same shape as any high-throughput backend you've built — you're just adding token/cost as first-class metrics.

### Defense checklist
- "I can swap model providers via config with no code change" — and you can show the interface.
- "Every request logs token counts and cost; I can query total spend by day and endpoint."
- "Cache hits return in <50ms at zero cost" — with a number.
- "Under rate limiting, the system degrades gracefully instead of cascading" — proven with a load test.

---

## Phase 1 — Prompt engineering as engineering, with evaluation (Weeks 2–3)

### The mental model
Juniors think prompting is about clever wording. Seniors think prompting is about **building a measurement system around an unreliable component**. The prompt is your only control surface over a model you can't retrain — so it must be versioned, tested, and regression-checked like any other critical code. The deliverable that wins interviews here is not a clever prompt. It's an **eval harness**. Say that out loud until it feels true.

### Concepts & vocabulary

**Prompt engineering.** The practice of structuring input to reliably get good output. The legitimate techniques (not magic incantations):
- **System prompt** — a high-priority instruction block that sets the model's role, constraints, and output contract for the whole conversation. Your most powerful lever.
- **Few-shot prompting** — including a handful of input→output *examples* in the prompt so the model infers the pattern. ("Zero-shot" = no examples; "few-shot" = a few.) Examples often beat paragraphs of instruction.
- **Output contract** — explicitly specifying the exact format you want back, ideally with a schema. Vague asks get vague (and varied) answers.
- **Delimiters** — using clear markers (XML tags, triple backticks) to separate instructions from data, which reduces the model confusing one for the other and improves *prompt-injection* resistance (more in Phase 5).

**Chain-of-thought (CoT).** Prompting the model to "think step by step" — to generate intermediate reasoning before its final answer. This measurably improves performance on reasoning-heavy tasks, because the model uses its own generated tokens as a scratchpad. The cost: more output tokens = more money and latency. Senior judgment is knowing when the accuracy gain justifies the cost.

**Self-critique / reflection.** Having the model review and revise its own output ("Here's a draft; now critique it and improve it"). Sometimes helps, sometimes just burns tokens. You evaluate, not assume.

**Evaluation ("evals") — the heart of the phase.** An *eval* is a systematic, repeatable measurement of how good your LLM system's outputs are. Without evals you are flying blind: you change a prompt, it "feels" better, and you've silently regressed three other cases. Key sub-concepts:
- **Golden dataset** — a curated set of inputs paired with known-good outputs (or human judgments). Your ground truth.
- **Exact-match / rule-based scoring** — for tasks with objectively correct answers (classification, extraction). Cheap, deterministic, unambiguous.
- **LLM-as-judge** — using a (often stronger) LLM to *score* another model's output against a rubric, for subjective qualities like helpfulness or tone where exact-match doesn't apply. Powerful but biased — judge models have known quirks (they favor longer answers, favor their own family's style), so you **calibrate** the judge against human-labeled samples before trusting it. Being able to discuss LLM-as-judge *and its biases* is a strong senior signal.
- **Pairwise comparison** — instead of scoring outputs absolutely, asking "is A or B better?" Often more reliable than absolute scores because relative judgment is easier than calibrated grading.
- **Regression testing** — running your eval suite automatically (in CI) so a change that worsens quality *fails the build* before it ships.

### The project — "Eval Harness + Prompt Registry"
Pick a real task with measurable output — support-ticket triage (classify into categories) is ideal because it has objective ground truth. Build:
- A **prompt registry** in Postgres: prompts are versioned rows, not hardcoded strings. You can diff versions and roll back.
- A **golden set** of 100+ labeled examples.
- An **eval runner** that, for any prompt version, reports accuracy, average cost, and p95 latency. (*p95 latency* = the latency the slowest 5% of requests experience — far more honest than an average, because tail latency is what users complain about.)
- An **LLM-as-judge** for any subjective dimension, with a human-checked calibration sample so you know the judge agrees with humans.
- **CI integration** (GitHub Actions): a prompt change that regresses accuracy below threshold fails the build.

Do this by hand once before reaching for tools like `promptfoo` — building the runner yourself is how the concepts move from vocabulary into understanding.

### Stack & rationale
Postgres holds the registry and golden set (durable, queryable, diffable). Pytest drives the runner because evals *are* tests conceptually. GitHub Actions gates merges. The point of versioning prompts in a DB rather than code is that non-engineers can eventually contribute prompts, and you get an audit trail of what changed when quality moved.

### Defense checklist
- "I never ship a prompt change without running it through an eval harness — here's mine."
- "A deliberately worse prompt fails my CI" — demonstrable.
- "I can state the accuracy-vs-cost tradeoff between a small and large model on my task, with numbers."
- "I understand LLM-as-judge's biases and I calibrate it against human labels."

---

## Phase 2 — RAG architecture & vector databases (Weeks 4–6)

### The mental model
This is the single most employable skill set in the document, so we spend three weeks. Most RAG in the wild is naive ("embed everything, grab top-5, stuff into prompt") and produces mediocre, hallucination-prone results. Senior value lives entirely in **retrieval quality** and **the ability to measure it**. If you take one thing from this whole document into an interview, make it the ability to talk rigorously about RAG.

### Concepts & vocabulary

**RAG (Retrieval-Augmented Generation).** A pattern where, instead of relying on what the model memorized during training, you *retrieve* relevant documents at query time and inject them into the prompt as context, then ask the model to answer *using that context*. Why it exists: it grounds answers in your own, current, authoritative data; it lets the model "know" things it was never trained on; and it gives you citations. RAG is not a model — it's an architecture around a model.

**Embedding.** A *vector* (a list of numbers, e.g. 1,536 of them) that represents the *meaning* of a piece of text, produced by an embedding model. The crucial property: texts with similar meaning produce vectors that are close together in this high-dimensional space, even if they share no words. "How do I reset my password" and "I forgot my login credentials" land near each other. This is what makes *semantic search* possible — searching by meaning rather than keyword.

**Vector / vector space / similarity metric.** The embedding lives in a *vector space*. To find "similar" text you measure distance between vectors, most commonly with **cosine similarity** (the cosine of the angle between two vectors — close to 1 means very similar direction/meaning). Retrieval = "find the stored vectors closest to my query vector."

**Vector database / vector index.** Storage optimized for "find the nearest vectors" queries. Searching exhaustively through millions of vectors is slow, so vector DBs use **ANN (Approximate Nearest Neighbor)** algorithms — they trade a tiny bit of accuracy for enormous speed. Two index types you must know:
- **HNSW (Hierarchical Navigable Small World)** — a graph-based index, very fast queries, higher memory use. The common default.
- **IVFFlat (Inverted File with Flat compression)** — partitions vectors into clusters; cheaper memory, slightly different speed/recall tradeoff.

The thing a senior articulates: ANN gives you a **recall vs latency vs memory** tradeoff, and you pick a point on that curve deliberately. **`pgvector`** is the Postgres extension that adds vector columns and these indexes — the pragmatic default because it keeps your vectors next to your relational data instead of running a separate system.

**Chunking.** Splitting source documents into smaller pieces ("chunks") before embedding, because you can't embed a whole 50-page doc as one vector meaningfully and you don't want to stuff 50 pages into the context window. *How* you chunk silently determines answer quality — chunk too small and you lose context, too large and retrieval gets imprecise. Strategies:
- **Fixed-size** — every N tokens. Simple, dumb, often splits mid-thought.
- **Recursive** — split on natural boundaries (paragraphs, then sentences) to stay under a size limit.
- **Semantic** — split where the topic actually shifts.
- **Structure-aware** — respect the document's own structure (headings, sections, code blocks).

Being able to say "I chose recursive chunking with overlap and here's the retrieval-precision number that justified it" is a senior tell.

**Chunk overlap.** Including a little of the previous chunk's tail at the start of the next, so a fact spanning a boundary isn't lost. A small, important detail.

**Hybrid search.** Combining **dense retrieval** (embeddings — good at meaning) with **sparse retrieval** (keyword methods like **BM25**, a classic ranking function — good at exact terms, names, IDs, codes). Pure vector search infamously misses exact matches ("error code X1492" — semantics don't help you there). You fuse the two rankings, often with **RRF (Reciprocal Rank Fusion)** — a simple, robust method that combines ranked lists by summing reciprocals of ranks. Hybrid beats pure-vector on most real corpora, and proving that with numbers is gold.

**Re-ranking / cross-encoder.** After retrieving a candidate set (say top-20) cheaply, you run a more expensive, more accurate model — a **cross-encoder** — that looks at the query and each candidate *together* and re-scores them, so the best few rise to the top. (The distinction: the embedding model encodes query and document *separately*; a cross-encoder encodes them *jointly*, which is more accurate but too slow to run over the whole corpus — hence the two-stage retrieve-then-rerank pattern.)

**Retrieval evaluation.** You evaluate retrieval *separately* from generation, because if retrieval feeds garbage in, no prompt can save the answer. Metrics:
- **Precision@k** — of the top-k retrieved chunks, what fraction are actually relevant.
- **Recall@k** — of all the relevant chunks that exist, what fraction made it into the top-k.
- **MRR (Mean Reciprocal Rank)** — how high up the first relevant result tends to appear.

And end-to-end, you measure **faithfulness** (does the answer actually follow from the retrieved context, or did the model invent?) and **answer relevance** (does it address the question?). The `ragas` library packages these. *Note your classical-ML background pays off here — precision and recall are the same concepts you already know, applied to retrieved chunks instead of classifier predictions.*

**Failure modes you must be able to name.** *Hallucination from thin context* (too little retrieved, model fills gaps by inventing). *Lost-in-the-middle* (models attend less to content buried in the middle of a long context — so ordering matters). *Stale data* (your index is out of date). *Citation drift* (the answer cites a source that doesn't actually support it).

### The project — "Production RAG over a real corpus"
Ingest a non-trivial corpus (company docs, a large public dataset, API documentation). Build the full pipeline:
1. **Ingestion + chunking** (recursive, with overlap) → **embedding** into `pgvector`.
2. **Hybrid retrieval**: `pgvector` (dense) + Postgres full-text search (sparse) fused with **RRF**.
3. A **cross-encoder re-rank** stage over the fused candidates.
4. **Answer synthesis with inline citations** pointing at the exact source chunk.
5. A **retrieval eval suite** (precision@k / recall@k on a labeled query set) *and* an end-to-end **faithfulness** eval.
6. **Redis caching** of embeddings and frequent queries.
7. A FastAPI `/query` endpoint streaming cited answers.

### Stack & rationale
`pgvector` over a dedicated vector DB (Pinecone, Weaviate) because keeping vectors in Postgres means one system, transactional consistency with your metadata, and it scales further than people assume — and you can articulate *when* you'd outgrow it (very high vector counts / QPS). Postgres FTS gives you the sparse half without adding infrastructure. A cross-encoder (Cohere Rerank, or a local one) for the quality jump. `ragas` so you're not hand-rolling faithfulness metrics.

### Defense checklist
- "Hybrid + re-rank beat naive vector search on my labeled set" — with the precision/recall numbers.
- "Every answer carries citations that point at the actual source chunk."
- "I can defend my chunking strategy and prove it beat an alternative."
- "I measured p95 query latency and it meets a target I set and can justify."
- "I evaluate retrieval separately from generation, because retrieval quality caps answer quality."

---

## Phase 3 — Your first real agent (Weeks 7–8)

### The mental model
Now the system *acts*, not just answers. Here's the senior inversion of the hype: the skill is **reliability and control, not autonomy**. A flashy agent that fails 30% of the time is a junior toy. A tightly-bounded agent that fails 2% and degrades safely is a senior artifact. Demystify it early: an agent is a `while` loop around tool-calling. Once you see that, the magic becomes engineering.

### Concepts & vocabulary

**Agent.** An LLM operating in a loop with the ability to take actions. The canonical loop: **reason** (the model decides what to do) → **act** (it calls a tool) → **observe** (it sees the tool's result) → repeat, until it decides it's done or hits a limit. The LLM is the decision-maker; the tools are its hands.

**Tool / function calling.** The mechanism by which an LLM invokes your code. You describe available tools to the model as schemas (name, description, parameters); when the model wants to use one, it returns a structured request naming the tool and its arguments; *your* code executes it and feeds the result back. The model never runs code itself — it *requests* calls, you execute them. This boundary is a security and reliability seam you control.

**ReAct (Reasoning + Acting).** A foundational agent pattern where the model explicitly alternates between reasoning traces ("I need the user's order total, so I'll query the DB") and actions (the actual query). Interleaving reasoning with action improves reliability and, crucially, gives you a *legible trace* of the agent's "thinking" for debugging.

**Memory (three kinds — keep them distinct).**
- **Short-term / conversational** — the running dialogue, held in the context window.
- **Working / scratchpad** — intermediate results the agent jots down mid-task.
- **Long-term** — knowledge persisted across sessions, typically retrieved via RAG. Note that "agent memory" is, under the hood, mostly *clever management of what goes into the finite context window* — there's no mystical memory, just retrieval and state.

**Control mechanisms — the senior heart of this phase.** Because an agent loops and acts, an unbounded one can spin forever, spend unboundedly, or take destructive actions. You impose:
- **Max iterations** — a hard cap on loop turns.
- **Token / cost budget** — stop if spend exceeds a ceiling.
- **Timeouts** — wall-clock limits.
- **Halting conditions** — clear definitions of "done."
- **Human-in-the-loop (HITL)** — a required human approval gate before any *state-changing* or irreversible action (sending an email, writing to a DB, spending money). The senior instinct: the agent can *propose* freely but can only *commit* through a gate.

**Idempotency in tools.** Design tools so that accidentally calling them twice doesn't double-charge a card or send two emails — because models *will* occasionally repeat a call. You know idempotency from API design; it's load-bearing here.

**Observability / tracing.** Recording every step the agent took — each reasoning step, each tool call and its arguments, each observation, tokens and latency per step — so a failed run can be *replayed* and understood. In agentic systems this isn't optional; a non-traceable agent is undebuggable.

### The project — "Bounded research/ops agent"
An agent with 3–5 *real* tools: your Phase 2 RAG endpoint as a `search` tool, a calculator, a read-only SQL tool against Postgres, an HTTP fetch. It plans, calls tools, and returns a cited result. Make it senior-grade with:
- **Hard budgets** — max steps, max tokens, max wall-clock — proven by tests that deliberately push the limits.
- **Pydantic validation** of every tool argument, so a hallucinated argument is caught at the boundary, not executed.
- A **full execution trace** persisted to Postgres that you can replay.
- A **human-approval gate** before any state-changing tool.

Build the loop yourself with the raw provider SDK. **Do not reach for a framework yet** — you cannot debug LangGraph if you've never written the loop it abstracts.

### Stack & rationale
Raw SDK tool-calling so the loop is *yours* and fully legible. Postgres for the replayable trace. Redis for conversational/working memory (fast, ephemeral). The whole design philosophy: maximize control and observability, minimize cleverness.

### Defense checklist
- "My agent provably never exceeds its step/token/time budget" — with the tests that force each limit.
- "Every run produces a replayable trace I can walk you through."
- "Malformed tool arguments are caught and recovered from, not crashed on."
- "I can explain when I'd use an agent versus a fixed chain for the same task" — and often the answer is the chain.

---

## Phase 4 — Multi-agent orchestration (Weeks 9–11)

### The mental model
Only now do we introduce an orchestration framework, and we introduce it **skeptically**. The interview-relevant wisdom is knowing when multi-agent genuinely helps (truly parallel or specialized subtasks) versus when it's expensive theater (most of the time — it can multiply your token cost 5–10× for marginal gain). Build it, measure it honestly, and be ready to argue *both sides*. The ability to argue against your own design is the clearest senior signal in this entire document.

### Concepts & vocabulary

**Multi-agent system.** Multiple specialized agents collaborating on a task, instead of one generalist agent doing everything. The bet is that specialization (a "researcher," a "writer," a "critic") plus division of labor beats a single agent juggling everything in one context window.

**Orchestration.** The coordination layer deciding which agent runs when, what information passes between them, and how results combine. Common **topologies**:
- **Supervisor / worker (orchestrator-worker)** — one coordinating agent routes subtasks to specialist workers and assembles their outputs. The most common and most defensible pattern.
- **Sequential pipeline** — agents in a fixed chain, each consuming the previous one's output.
- **Parallel fan-out / gather** — split work across agents simultaneously, then merge. Genuinely useful when subtasks are independent.
- **Debate** — agents argue opposing positions to surface a better answer. Occasionally powerful, often just costly.

**Handoff & shared state.** How agents pass context to one another. The naive approach (forward the entire conversation to each agent) causes **quadratic token blowup** — cost scaling badly as agents and context grow. Seniors manage handoffs deliberately: pass *summaries* or *structured state*, not raw history. Where that shared state lives (Postgres for durability, Redis for fast coordination) is your call.

**Coordination failure modes — name these in interviews.**
- **Deadlock** — agents waiting on each other, nothing progresses.
- **Infinite delegation** — agents passing a task back and forth without resolving it.
- **Context loss across handoffs** — critical detail dropped in translation between agents.
- **Compounding error** — each agent has some error rate; chain enough of them and reliability collapses multiplicatively. This is *the* core argument against gratuitous multi-agent designs, and you should be able to state it crisply. (Example: five agents each 95% reliable end-to-end is 0.95⁵ ≈ 77% — the math is brutal and worth memorizing.)

**Framework landscape.** **LangGraph** (models the system as a graph of nodes and state — explicit, debuggable), **CrewAI** (role-based agents, higher-level, faster to start), **AutoGen** (conversation-centric multi-agent), and the **Anthropic / OpenAI SDK agent primitives** (lower-level, maximal control). Each trades control for convenience. Pick one, *justify the pick*, and know what it hides from you.

### The project — "Supervisor multi-agent system"
Choose a task that *genuinely* decomposes — e.g. "research a topic → draft a report → fact-check it → format it." Implement a **supervisor** routing to specialists (researcher = your Phase 2 RAG agent; writer; critic). Persist shared state in Postgres; use Redis for inter-step coordination. Then do the part that earns the title: run an honest **eval comparing the multi-agent system against a single well-prompted agent** on quality, cost, and latency — and write up exactly when each wins.

### Stack & rationale
Pick LangGraph *or* the SDK primitives — and be able to say why (LangGraph for explicit state-graph debuggability; SDK primitives for maximal control). Reuse your Phase 1 eval harness, now extended to whole-workflow comparison. The deliverable's spine is the comparison, not the system itself.

### Defense checklist
- "The system completes the decomposed task with persisted, inspectable state."
- "I have a numbers-backed comparison against the single-agent baseline on quality, cost, and latency."
- "I can name the specific coordination failure modes I hit and how I bounded them."
- "I can argue *against* my own multi-agent design" — the senior tell.

---

## Phase 5 — Production deployment & hardening (Weeks 12–14)

### The mental model
Take your strongest artifact — the RAG system or the agent — all the way to production-grade. This is where your existing AWS and backend seniority becomes a genuine competitive advantage: most GenAI candidates *cannot* operate these systems, and you already can. Lean into it hard in interviews.

### Concepts & vocabulary

**Serving topology on AWS.** The production shape: **ECS Fargate** (run containers without managing servers), **RDS** (managed Postgres + pgvector), **ElastiCache** (managed Redis), **ALB** (Application Load Balancer — distributes traffic), **S3** (object storage for documents/artifacts), **CloudWatch** (metrics, logs, alarms). **Bedrock** is AWS's managed gateway to multiple foundation models if you want models inside your AWS security boundary.

**LLM-specific production concerns.** Standard web ops, plus quirks: **streaming through a load balancer** (ALBs and timeouts fight long-lived streaming connections — you must configure for it); **long-request timeouts** (LLM calls are slow; default timeouts kill them); **per-user token-budget rate limiting** (stop one user from running up your bill); **semantic caching** (cache by *meaning* — if a new query is semantically near a cached one, serve the cached answer — a step beyond exact-match caching).

**Observability for LLM systems.** Beyond standard metrics: **tracing** every LLM/agent step (tools like **Langfuse** or **Arize Phoenix**, or OpenTelemetry), dashboards for **latency / cost / quality**, **quality monitoring on live traffic** (sampling real outputs and scoring them — your evals don't stop at deploy), and **drift detection** (noticing when production inputs or output quality wander from what you tested — a concept you already know from classical ML, applied to live LLM traffic).

**Safety & guardrails.** The defensive layer:
- **Prompt injection** — an attack where malicious text in user input (or in a *retrieved document* — the scarier vector) hijacks the model's instructions ("ignore previous instructions and..."). The RAG-borne version is subtle: you trust your documents, but a poisoned one can carry an injection. Defenses: strong delimiting, treating retrieved content as untrusted data, instruction hierarchies, output validation.
- **Output filtering** — catching toxic, off-policy, or sensitive content before it reaches users.
- **PII handling** — detecting and protecting personally identifiable information.
- **Jailbreak resistance** — hardening against attempts to bypass the model's safety behavior.

**Cost controls.** **Semantic + exact caching**; **model routing** (try a cheap model first, escalate to an expensive one only when confidence is low — often cuts cost dramatically); **budget alarms** (CloudWatch alerts before spend surprises you).

**CI/CD with eval gates.** The capstone idea tying it all together: your deployment pipeline runs the eval suite from Phase 1 and **blocks the deploy if quality regresses**. This is the single most senior thing you can demonstrate — treating LLM quality as a release gate, exactly like tests gate normal code.

### The project — "Ship it"
Deploy your best artifact to AWS, fully containerized, behind an ALB, with:
- **IaC** (Infrastructure as Code — CDK or Terraform — your whole stack defined in version-controlled code, not clicked together in a console).
- A **dashboard** (CloudWatch + Langfuse/Phoenix) showing p50/p95 latency, cost per request, and a live quality metric.
- **Per-user rate limiting** in Redis.
- **Prompt-injection and output guardrails**, with tests that actively try to break them.
- A **CI pipeline** running your eval suite and **blocking deploys on regression**.
- A **load test** with documented results.
- A one-page **architecture doc** plus a "what I'd do at 100× scale" section.

### Stack & rationale
This phase is intentionally where you flex existing strengths — IaC, load balancers, containers, dashboards are your home turf. The new surface area is *LLM-specific*: streaming through the ALB, semantic caching, eval-gated deploys, injection defense. That combination — seasoned ops engineer *plus* LLM-specific production knowledge — is exactly the rare profile these companies hire.

### Defense checklist
- "There's a public URL serving my system with streamed responses."
- "I have a dashboard showing latency, cost, and live quality on real traffic" — screen-shareable.
- "Here's a documented prompt-injection attempt my guardrails catch."
- "Deploys are blocked when evals regress" — demonstrated.
- "I can verbally scale this architecture to 100× and tell you what breaks first."

---

## Putting it together — how to defend the whole thing

By the end you have five deployed, eval-backed artifacts. But understand the meta-point: **at senior level, the artifacts are evidence, not the argument. The argument is your judgment.** When interviewing at Anthropic, OpenAI, Amazon and peers, lead with these themes — each is a sentence you should be able to expand into ten minutes:

1. **Evaluation discipline.** "I don't ship LLM changes without an eval harness, and I gate deploys on it." This separates senior from junior more than any other single thing. Most people cannot say it and mean it.
2. **Retrieval quality.** Defend chunking, hybrid search, and re-ranking with measured numbers. RAG is everywhere; rigor about it is rare.
3. **Reliability & bounding.** Budgets, halting, graceful degradation, replayable traces. You make unpredictable components behave predictably.
4. **Cost engineering.** Caching, model routing, token accounting. Speak fluently in dollars-per-request — most candidates can't.
5. **Knowing when *not* to.** When a chain beats an agent; when single-agent beats multi-agent. Senior engineers are trusted precisely because they don't build the complicated thing reflexively.
6. **Production operation.** Your AWS background means you can actually run these systems. This is your moat — most GenAI talent can't operate what they build.

## How this connects to the classical ML you're already doing

Your ongoing ML work is *complementary*, not redundant. Classical ML gives you the statistical literacy to reason about evaluation properly — you already understand precision/recall, overfitting, and distribution shift, and those map directly onto eval design, retrieval metrics, and drift detection here. The applied-GenAI track in this document is a *different layer of the stack*: you're not training models, you're architecting reliable systems *around* them. The strongest candidates move fluidly between both — they can discuss an embedding model's training intuition *and* deploy a cost-bounded RAG service. Keep both plates spinning; the combination is rarer than either skill alone, and it's exactly the profile a place like Anthropic or OpenAI values in an engineer who has to reason about models *and* ship the systems that serve them.

## What this timeline deliberately omits (so you can speak to scope honestly)

- **Fine-tuning, training, RLHF** — adjusting model weights is a different track and rarely the senior *application* engineer's day job. (*RLHF = Reinforcement Learning from Human Feedback, the technique used to align models to human preferences during training.*) If a target role emphasizes it, bolt on a dedicated phase; ask me and I'll write one.
- **Deep transformer internals** — you should be able to explain attention and next-token prediction conceptually (covered in the vocabulary section), but implementing a transformer from scratch is a research-track investment, not an applied-engineering one.
- **Framework-first habits** — frameworks are introduced late and skeptically on purpose. Building the loops by hand first is what lets you debug the frameworks later.

---

*Use this as a living document. Where you're already strong, compress and reinvest the time in evaluation and retrieval — that's where the senior signal concentrates. Where a term still feels slippery, that's your flag to build the smallest possible thing that makes it break, because you'll never forget a concept you've watched fail.*
