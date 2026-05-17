# Quick AI RAG Chat Web App

A small, hands-on project for learning **Retrieval-Augmented Generation (RAG)**. Two Gradio chat apps that recommend GPUs from a tiny vector database, with semantic search, metadata filtering, and (in the improved version) Pydantic-structured LLM output and agentic tool calling.

The data in the ChromaDB vector database aren't 100% accurate. Just quick sample data I used for this project. You can use cars, food, airplanes, cities, etc.

Built with:
- **[ChromaDB](https://www.trychroma.com/)** — local vector database (embeddings handled for you)
- **[Google Gemini](https://ai.google.dev/)** — LLM for re-ranking and reasoning
- **[Gradio](https://gradio.app/)** — chat UI
- **[Pydantic](https://docs.pydantic.dev/)** *(improved version only)* — typed/validated LLM output

---

## Repo Contents

| File | What it is |
|------|------------|
| [`gradio_app.py`](gradio_app.py) | **Original chat app.** RAG retrieval with hard filters + LLM re-ranking. Returns prose. Minimal moving parts — easiest to read. |
| [`gradio_app_pydantic.py`](gradio_app_pydantic.py) | **Improved chat app.** Same RAG flow + Pydantic structured output + Gemini tool calling (`get_live_price`) + a parallel "structured view" panel (markdown card + sortable dataframe) populated from the validated Pydantic object. |
| [`.env.example`](.env.example) | Template for API keys. Copy to `.env` and fill in. |

---

## Quick Start

### 1. Clone and create a virtual environment

```bash
git clone <this-repo-url>
cd rag
python3 -m venv .venv
source .venv/bin/activate   # On Windows: .venv\Scripts\activate
```

### 2. Install dependencies

There's no `requirements.txt` yet. Install directly:

```bash
pip install gradio chromadb google-genai pydantic python-dotenv
```

Optional (only if you want to use Cohere for embeddings instead of Chroma's built-in default):

```bash
pip install cohere
```

### 3. Configure your API key

Copy the template and fill in your Gemini API key (free tier available at [aistudio.google.com/apikey](https://aistudio.google.com/apikey)):

```bash
cp .env.example .env
```

Edit `.env`:

```env
GEMINI_API_KEY=your_actual_key_here
# Optional:
# cohere=your_cohere_key_here
```

### 4. Run an app

The original chat app:

```bash
python gradio_app.py
```

The improved Pydantic + tools version:

```bash
python gradio_app_pydantic.py
```

Either will open at <http://127.0.0.1:7860>.

---

## Using the Chat App

Once the Gradio UI is open:

1. Adjust the sliders on the left — **Max Price**, **Min VRAM**, **Max Power**, and brand checkboxes. These are *hard filters* applied at retrieval time.
2. Type a question in the chat box, e.g.:
   - *"What's the best card for AI/ML work?"*
   - *"Find me the GPU with the most VRAM for the least amount of money."*
   - *"I want a quiet card for office work."*
3. Press **Send** (or hit Enter in the improved app).
4. The model retrieves matching GPUs from Chroma, re-ranks them based on your specific question, and returns a recommendation with a top pick, runners-up, and reasoning.

In `gradio_app_pydantic.py` you'll also see a structured-view panel below the chat with a styled card for the top pick and a sortable dataframe for the runners-up — both populated directly from the validated Pydantic object.

---

## How It Works

### The RAG flow (both apps)

```
User question + slider filters
            │
            ▼
    Chroma semantic search
    (query embedding vs. GPU
     description embeddings)
            │
            ├── hard filters (price, VRAM, power, brand) applied via `where=`
            │
            ▼
    Top-k results with
    distance-based confidence
            │
            ▼
    Prompt sent to Gemini:
    "Here are the candidates,
     re-rank to answer the user"
            │
            ▼
    LLM reply → chat
```

The confidence score per result is computed as `(1 - distance/2) * 100`, mapping Chroma's distance metric onto a 0–100 similarity score.

### What the improved version adds

`gradio_app_pydantic.py` keeps the entire RAG flow from the original, layered with:

1. **Pydantic schema for the LLM response.** `Recommendation` and `GPUPick` define the exact shape the model must produce. The Gemini call sets `response_schema=Recommendation`, and `response.parsed` returns a validated Pydantic object instead of free-form prose.
2. **Confidence override.** After parsing, the displayed confidence on each pick is replaced with the actual Chroma-derived score, so what you see in the UI is the real similarity number — not whatever the LLM stamped on the field.
3. **Agentic tool calling.** A `get_live_price(model)` Python function is exposed via `tools=[get_live_price]`. Gemini's automatic function calling decides if and when to invoke it. The prompt frames it as optional to avoid unnecessary round-trips.
4. **Structured view panel.** Below the chat, a `gr.Markdown` card and `gr.Dataframe` are populated from the Pydantic object. The dataframe needs typed columns (int for price/VRAM, float for confidence) — exactly what Pydantic guarantees.
5. **Enriched embedding query.** The slider preferences are baked into the query text sent to Chroma, so retrieval similarity reflects preference-fit, not just keyword overlap.
6. **Adaptive prompt.** The user's question is included in the LLM prompt explicitly, and the re-rank rule is query-driven ("answer the user's specific question") rather than hardcoded to "maximize VRAM at budget."

---

## Customizing the Data

The GPU dataset is seeded inline in both Gradio apps (look for the `collection.upsert(...)` block near the top). To add your own items:

1. Add a new `id`, `document` (free-text description), and `metadata` entry.
2. Make sure the metadata fields you filter on (e.g. `price`, `vram`, `power`, `brand`) are numbers/strings matching what the sliders use.
3. Restart the app.

For persistent storage across runs, swap `chromadb.Client()` for `chromadb.PersistentClient(path="./chroma_db")`.

---

## Troubleshooting

### "Unknown model" error
The default model is `gemini-3-flash-preview`. If Google retires or renames it, you'll see an SDK error. Swap to a current model ID (e.g. `gemini-2.5-flash`) — change the `MODEL` constant near the top of `gradio_app_pydantic.py`, or the inline `model=` arg in `gradio_app.py`.

### Enter key doesn't submit in the chat box
In Gradio 6, multi-line `Textbox` (`lines>1`) treats Enter as a newline by default. The improved app works around this with a small JavaScript handler that listens at the document level (passed via `demo.launch(js=...)`). If you've modified that section, make sure:
- The `js=` parameter is passed to `launch()`, **not** to `gr.Blocks()` (Gradio 6 moved it).
- The JS string is wrapped in an IIFE (`(() => { ... })()`) so it self-executes.

### App fails to launch with `gr.Chatbot(type="messages")`
Don't pass `type="messages"` in Gradio 6 — it's the default and explicit passing can break startup in some versions.

### `Warning: there are non-text parts in the response: ['function_call']`
Informational, not an error. Fires when the response contains both a tool-call event and the final JSON. Safe to ignore — recommendations still come through correctly.

### Response feels slow
The biggest latency contributor is tool calling — each invocation of `get_live_price` adds a round-trip to Gemini. The current prompt makes it opt-in, so most queries are still single-call. If you want maximum speed, remove `tools=[get_live_price]` from the `GenerateContentConfig`.

---

## Dependencies (tested versions)

| Package | Version |
|---------|---------|
| `gradio` | 6.14.0 |
| `chromadb` | 1.5.9 |
| `google-genai` | 2.3.0 |
| `pydantic` | 2.13.4 |
| `python-dotenv` | 1.2.2 |
| `cohere` *(optional)* | 5.21.1 |

Python 3.10+ recommended (3.13 tested).

To freeze your own pinned set after install:

```bash
pip freeze > requirements.txt
```

---

## Roadmap / Ideas

- Replace `get_live_price` mock with a real price-lookup API.
- Persist Chroma data across runs (`PersistentClient`).
- Add a "Compare" button that takes two `GPUPick`s and renders a side-by-side table.
- Log each recommendation to CSV/SQLite for analytics over time.
- Swap embedding model via Cohere (the `.env` slot is already there).
- Extend the dataset beyond 8 cards or load from a CSV at startup.

---

## Notes on the Project

This started as a learning exercise to understand how RAG, structured LLM output, and tool-calling actually fit together in a small app. The two versions are kept side-by-side intentionally — the original is the simplest possible working RAG chat, and the improved version layers in everything else without changing the core retrieval flow.
