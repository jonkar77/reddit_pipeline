# Reddit Pain-Point Research Pipeline

Finds trading communities on Reddit, mines them for concrete problems, and ranks the recurring ones as software opportunities. A local LLM (Ollama / Qwen) does extraction only; scoring is rule-based and reproducible.

```
 PRE-FETCH LAYER                                       PIPELINE
 discover ─► classify ─► review ─► approved ─►  fetch ─► process ─► triage ─► comments ─► analyze ─► opportunities ─► export
 (search,    (rules,     (you, in  subreddits   (raw     (clean,   (cheap    (survivors  (full LLM   (cluster across    (CSV)
 snowball,   activity,    a CSV)    table        store)   dedupe)   keep/     only)       extraction) communities,
 seeds)      local LLM)                                              drop)                            score)
```

Everything lives in one SQLite file (`data/pipeline.db`). Each stage reads one layer and writes the next, so any stage can be re-run without touching earlier ones, and nothing re-hits Reddit to reprocess.

## Setup (macOS / zsh)

```zsh
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
ollama pull qwen3.5:9b
ollama pull nomic-embed-text
cp .env.example .env          # put REDDIT_COOKIE in .env (never in code)
```

## Quick start

**1. Build your list of communities (once, then occasionally refresh)**

```zsh
python -m reddit_pipeline discover --seed algotrading,options,thetagang   # search + snowball + your seeds
python -m reddit_pipeline classify                                        # rules, activity check, local-LLM verdict
python -m reddit_pipeline review-export                                   # -> output/subreddit_review.csv
#   open the CSV, type approve / reject in the `decision` column, save, then:
python -m reddit_pipeline review-apply output/subreddit_review.csv
python -m reddit_pipeline subreddits --status approved                    # check the result
```

**2. Run the pipeline over every approved community**

```zsh
python -m reddit_pipeline fetch                  # incremental; --max-subs 5 for a small first run
python -m reddit_pipeline process
python -m reddit_pipeline triage --limit 50      # cheap title+body pass; drop --limit for everything
python -m reddit_pipeline comments --limit 200   # only for posts that survived triage
python -m reddit_pipeline analyze --limit 20     # small trial first, hand-check the output
python -m reddit_pipeline eval-sample --n 50     # label output/eval_sample.csv, then eval-score it
python -m reddit_pipeline analyze                # full run (resumable, cached)
python -m reddit_pipeline opportunities
python -m reddit_pipeline export                 # output/opportunities_ranked.csv + cluster_review.csv
```

Already have data? `import-json --posts-dir reddit-test-data --comments-dir analysis/comments` brings it in (those subreddits are registered as approved) and skips re-fetching. After approving or rejecting communities, run `process` so their posts are included or excluded.

Other: `stats`, `retry-failed`, `fetch --backfill`, `--subreddit NAME` on any command to restrict a run to one community, `--no-triage` to bypass the triage gate, `--backend coachwhip` for an OpenAI-compatible server. One-shot: `run-all --fetch`.

## The pre-fetch layer

| Step | What it does | Cost |
|---|---|---|
| `discover` | Keyword search (paginated, 18 trading terms by default, override with `--queries`), snowballing (r/xyz mentions and crosspost sources in data you already hold, plus links in approved communities' sidebars), and `--seed` for communities you name. Everything lands as `candidate` except seeds. | Reddit requests only |
| `classify` | Rules reject NSFW, quarantined, private, tiny, and obvious off-topic ("trading cards") communities. One profile request per survivor measures posting rate and recency, which drops dead communities. The local LLM then judges "is this about trading financial markets?" and picks a category. | 1 request + 1 LLM call each |
| `review-export` / `review-apply` | You approve or reject the LLM-confirmed candidates in a CSV. Nothing is fetched until a community is `approved`. `--all` re-opens decided ones. `classify --auto-approve` skips this step if you prefer. | A few minutes, once |

Every rejection stores a reason (`stats`, `subreddits --status rejected`), so you can see if a rule is too strict. Tune `min_subscribers`, `max_inactive_days`, `min_posts_per_week` and `snowball_min_mentions` in `config.py`.

## Stage contracts

| Stage | Reads | Writes | Guarantees |
|---|---|---|---|
| `discover` / `classify` | Reddit search, about, /new | `subreddits` | Candidates never fetched until approved; every rejection has a reason |
| `fetch` | approved subreddits | `raw_posts` | Upsert by Reddit ID, checkpoint per subreddit, `max_age_days` window, schema-checked, one bad community never stops the rest |
| `process` | `raw_posts` | `posts` | Normalize, dedupe, auditable filter reasons; rebuilt from raw every run |
| `triage` | `posts` | `triage` | Title+body only, tuned for recall, cached by input hash |
| `comments` | triaged-in posts | `raw_comments` | Best posts first, measured against each community's own median |
| `analyze` | posts + comments | `extractions` | Schema-validated JSON, retry with error feedback, cache by input hash, failures to a queue |
| `opportunities` | `extractions` | `runs`, `opportunities` | Clusters across communities, LLM names each, deterministic score |
| `export` | `opportunities` | CSVs | A view of one run; the DB stays the source of truth |

## Key design decisions

- **Cache key = hash(prompt version + model + prompt + context).** Change the prompt (bump `prompt_version`), the model, or a post's comments and exactly the affected posts are re-analysed.
- **Triage protects the scarce resources:** Reddit requests and local-model time. Comment fetching and full analysis happen only for posts a cheap pass kept.
- **Fair budgeting across communities.** Posts are prioritised by engagement relative to their own subreddit's median, so one huge community can't use up a `--limit`.
- **The LLM extracts, code scores.** Score = frequency (unique authors) + engagement (relative to each post's community) + intensity + graded willingness-to-pay evidence + **community spread** (how many distinct communities share the pain; one community scores 0 here). Weights are in `config.py`; every component is exported.
- **Cross-posts count once.** Identical text is de-duplicated across communities, so spread reflects genuinely separate discussions.
- **Humans decide which communities count; the model only recommends.**
- **Build difficulty and competition are manual columns** in the export; fill them in for your top ~20 only.
- **Pluggable edges:** `Transport` (curl+cookie today; swap in OAuth/PRAW), `LLMClient` (Ollama, OpenAI-compatible), `Embedder`.

## Evaluating changes

1. `eval-sample` writes a random sample with blank `label_has_pain` / `label_post_type` columns.
2. Fill in 50-100 rows by hand (`1`/`0` for has_pain).
3. After any prompt or model change: `analyze`, then `eval-score` for precision / recall / F1, logged in the `runs` table.
4. Review `cluster_review.csv` each run (it now shows each post's subreddit) and mark `cluster_member_ok`.
5. Triage is recall-tuned; spot-check what it rejected (`SELECT * FROM triage WHERE worth=0`) before trusting it at scale.

## Tests

```zsh
python -m unittest -v     # offline: stub LLM, stub embedder, fake Reddit (search, about, listings)
```

## Known limits

- Reddit listings stop at roughly 1000 posts, so `fetch --backfill` can't go deeper; run `fetch` regularly to accumulate history.
- Snowballing only finds communities that appear in data you already hold, so it improves as the dataset grows.
- Greedy clustering is order-dependent; `--cluster-method hdbscan` needs `scikit-learn`.
- Long threads are trimmed to the context budget, not split across calls.
- Scraping many communities with a session cookie is fragile and may conflict with Reddit's terms. Check their current data-use policy before scaling or building a product on it, and consider the official OAuth API (the fetcher sits behind a small `Transport` interface for exactly this).
