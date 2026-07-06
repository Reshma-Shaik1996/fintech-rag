# Fintech RAG — Grounded Q&A over Financial Documents

Production-style RAG system in C#/.NET 8 that answers questions about
JPMorgan's 2025 MD&A (117-page SEC filing) with inline page citations,
using hybrid retrieval (semantic + keyword search).

**Stack:** .NET 8 · Semantic Kernel · Gemini (chat + embeddings) · PdfPig

## Architecture

PDF → page extraction (PdfPig, content-ordered) → sentence-aware chunking
(1500 chars, 200 overlap; 377 chunks) → Gemini embeddings (768-dim, batched,
cached) → hybrid retrieval [cosine similarity + BM25, fused via Reciprocal
Rank Fusion] → grounded generation with inline citations + honest refusal

## Sample output

(screenshot: grounded answer with [p23-c2] citation, and the refusal case)

## Key engineering decisions & challenges

1. **Table extraction quality** — naive PDF text extraction scrambled financial
   tables; switching to content-ordered extraction restored usable structure.
2. **Embedding model deprecation mid-build** — text-embedding-004 was retired
   Jan 2026; migrated to gemini-embedding-001 with outputDimensionality=768.
   Lesson: embedding models are versioned dependencies.
3. **Rate limiting (429s)** — free-tier throttling handled with exponential
   backoff (5s→40s), smaller batches, and a local embedding cache so unchanged
   content is never re-embedded.
4. **RRF score ties** — when vector and BM25 disagree completely, rank fusion
   can tie a relevant and an irrelevant chunk. Mitigated by retrieving top-5
   and letting the grounded prompt filter; proper fix is measured evaluation.
5. **Known limitation** — granularity confusion: asked about 2025 credit losses, the system answered "$11.5 billion" with a valid citation — but that figure is the consumer segment provision; the firmwide total is $14.2 billion. Retrieval surfaced a true-but-partial context and generation over-generalized it. A precision-forced query ("firmwide provision for full year 2025") returns the correct $14,212 million. This failure mode — cited, plausible, wrong granularity — is undetectable without systematic evaluation, which motivated the companion eval-harness project.

## Run it

1. `dotnet user-secrets set "Gemini:ApiKey" "<your key>"` (in FintechRag.Console)
2. Download the JPMorgan 2025 MD&A PDF → `data/annual-report.pdf`
3. `dotnet run --project FintechRag.Console`
   (first run embeds ~377 chunks in batches; subsequent runs use the cache)

## Roadmap

- Evaluation harness with golden dataset (Project 2 — in progress)
- Reranking and semantic chunking, adopted only if evals show measurable gains
- Azure OpenAI provider swap (config-level change, provider-agnostic design)