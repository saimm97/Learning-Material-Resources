
# Debugging Summary: Polyfactory + SQLAlchemy FK Violations in Seed Script

**Subject:** Seeding a FastAPI eval/prompt registry database using `polyfactory`'s `SQLAlchemyFactory` to generate `PromptVersions`, `PromptFamily`, `GoldenExamples`, `EvalRunner`, and `EvalResults` rows. `EvalResults` has FKs to both `GoldenExamples` (loaded from JSON) and `EvalRunner`. The recurring failure was a Postgres `ForeignKeyViolation` on `eval_results.golden_examples_id` — the seed script kept inserting FK values pointing at non-existent golden examples.

---

**Error 1: FK violation — `Use(lambda: random.choice(golden_ids))` as kwarg**

Random IDs (5255, 2446, etc.) appearing in the INSERT. `Use` passed as a runtime kwarg to `create_batch_sync` isn't reliably resolved by polyfactory — it's meant for class-level field declarations. The wrapper was ignored and polyfactory's auto-generated random int for the FK column won.

**Error 2: Same violation — `Use(random.choice, golden_ids)` (correct signature)**

Different random IDs (9872, 2530, etc.), same root cause. Concept clarification: random ints weren't caused by `Use` — they came from polyfactory auto-generating values for FK columns inferred from the SQLAlchemy model. `Use` was the attempted override; it just wasn't landing at kwarg-level.

**Error 3: Same violation — plain int kwarg**

You dropped `Use` and passed `golden_examples_id=random.choice(golden_ids)` directly to `create_batch_sync`. Still random IDs. Suggested batch path also wasn't applying the kwarg.

**Real culprit: `__set_relationships__ = True`**

Traceback was firing at `EvalRunnerFactory.create_batch_sync(5)` — one line *above* the eval_results loop. That was the giveaway. With `__set_relationships__ = True`, creating an `EvalRunner` cascaded and auto-created related `EvalResults` rows with random FK values. Those bogus rows got flushed on commit → FK violation. Your explicit eval_results loop was never reached. Every "random ID" you'd been seeing came from cascade-generated rows, not your loop. Fix: set `__set_relationships__ = False` on all factories (critical on `EvalRunnerFactory` and `GoldenExamplesFactory` since those have `eval_results` as a child relationship).

**Error 4 (minor): `create_sync` size arg**

`EvalResultsFactory.create_sync(5, ...)` — `create_sync` doesn't take a batch size. That `5` became a positional model-constructor arg. Fix: `for _ in range(5)` loop, drop the `5`.

---

**Side notes covered:**

- `Base.metadata.drop_all()` drops tables + data, not just data — fine for seed scripts.
- `session.flush()` alone doesn't commit; needed explicit `session.commit()` in `fetch_golden_examples` rather than relying on the next factory's internal commit.
- `expire_on_commit=True` (SQLAlchemy default) expires ORM objects after commit; accessing `.id` triggers a reload — fine as long as rows are actually persisted.

**Meta lesson:** the traceback line pointed at `EvalRunnerFactory` from error #1, but the failing SQL was on `eval_results`. That mismatch was the diagnostic clue the entire time. When the failing SQL table doesn't match the code line in the traceback, look for cascading side effects — that's where the bug lives.
