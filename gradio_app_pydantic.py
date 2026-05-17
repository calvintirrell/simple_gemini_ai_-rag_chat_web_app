import gradio as gr
import chromadb
from google import genai
from google.genai import types
from pydantic import BaseModel, Field, ValidationError
from dotenv import load_dotenv

# Load API Key
load_dotenv()

# --- SETUP ---
gemini_client = genai.Client()
chroma_client = chromadb.Client()
collection = chroma_client.get_or_create_collection(name="gpu_db3")
MODEL = "gemini-3-flash-preview"

# --- DATA SEEDING ---
collection.upsert(
    ids=["1", "2", "3", "4", "5", "6", "7", "8"],
    documents=[
        "AMD 7900XTX is the previous generation of GPU, current best overall value for gaming and AI/ML related work and best for compute intensive applications with 24gb of vram. This costs $400.",
        "AMD 6900XTX is the 2 generation prior of GPU, previous best overall value for gaming and some AI/ML related work and solid for compute intensive applications with 16gb of vram. This costs $300.",
        "AMD 5900XTX is the 3 generation prior of GPU, medium size  and solid gaming option along with office work and some AI/ML. This GPU with 8gb of vram costs $200. ",
        "AMD 4900XTX is the 4 generation prior of GPU, small size and uses less power. Good for some gaming and office work. This GPU with 4gb of vram costs $100.",
        "Nvidia 4090 is the previous generation of GPU, best overall gaming and AI/ML related work with 24gb of vram. This costs $600.",
        "Nvidia 3090 is the 2 generation prior of GPU, best overall for gaming and some AI/ML related work with 24gb of vram. This costs $500.",
        "Nvidia 2090 is the 3 generation prior of GPU, medium size  and solid gaming option along with office work and some AI/ML. This GPU with 8gb of vram costs $400. ",
        "Nvidia 2080 is the 3 generation prior of GPU, small size and uses less power. Good for some gaming and office work. This GPU with 8gb of vram costs $300."
    ],
    metadatas=[
        {"brand": "AMD", "model": "7900XTX", "vram": 24, "size": "big", "power": 9, "price": 400},
        {"brand": "AMD", "model": "6900XTX", "vram": 16, "size": "big", "power": 7, "price": 300},
        {"brand": "AMD", "model": "5900XTX", "vram": 8, "size": "medium", "power": 5, "price": 200},
        {"brand": "AMD", "model": "4900XTX", "vram": 4, "size": "small", "power": 3, "price": 100},
        {"brand": "Nvidia", "model": "4090", "vram": 24, "size": "big", "power": 9, "price": 600},
        {"brand": "Nvidia", "model": "3090", "vram": 24, "size": "big", "power": 8, "price": 500},
        {"brand": "Nvidia", "model": "2090", "vram": 12, "size": "medium", "power": 6, "price": 400},
        {"brand": "Nvidia", "model": "2080", "vram": 8, "size": "small", "power": 5, "price": 300}
    ]
)

# --- TYPED OUTPUT SCHEMA (Layer 1: structured output) ---
class GPUPick(BaseModel):
    model: str
    brand: str
    price_usd: int
    vram_gb: int
    confidence: float = Field(ge=0, le=100)
    reasoning: str

class Recommendation(BaseModel):
    top_pick: GPUPick
    runners_up: list[GPUPick]
    summary: str


# --- TOOL THE AGENT CAN CALL ---
# Gemini reads the type hints + docstring to build the JSON schema it
# shows the model. Mock for now — would hit a real price API in prod.
def get_live_price(model: str) -> str:
    """Look up the current (mock) live retail price for a specific GPU.

    Args:
        model: A GPU model identifier, e.g. "4090" or "7900XTX".

    Returns:
        A human-readable price string the model can quote in its reasoning.
    """
    all_rows = collection.get()
    needle = model.lower()
    for meta in all_rows["metadatas"]:
        name = meta["model"].lower()
        if needle in name or name in needle:
            return (
                f"Live retail price for {meta['brand']} {meta['model']}: "
                f"${meta['price']} (just now)."
            )
    return f"No live price found for {model}."


# --- STRUCTURED-VIEW RENDERERS ---
# Each consumes a Pydantic object directly. The Dataframe in particular
# can't be built from prose — it needs typed fields (price as int, VRAM
# as int, etc.), which is exactly what Pydantic guarantees.
def format_top_card(top: GPUPick) -> str:
    return (
        f"### Top Pick: {top.brand} {top.model}\n\n"
        f"| Spec | Value |\n"
        f"|---|---|\n"
        f"| Price | ${top.price_usd} |\n"
        f"| VRAM | {top.vram_gb} GB |\n"
        f"| Retrieval Confidence | {top.confidence}% |\n\n"
        f"**Why:** {top.reasoning}"
    )


def format_runners_rows(runners: list[GPUPick]) -> list[list]:
    return [
        [f"{p.brand} {p.model}", p.price_usd, p.vram_gb, p.confidence, p.reasoning]
        for p in runners
    ]


# --- RENDER A Recommendation AS CHAT MARKDOWN ---
def format_recommendation(rec: Recommendation) -> str:
    top = rec.top_pick
    lines = [
        f"## Top Pick: {top.brand} {top.model}",
        f"- **Price:** ${top.price_usd}",
        f"- **VRAM:** {top.vram_gb} GB",
        f"- **Retrieval Confidence:** {top.confidence}%",
        f"- **Why:** {top.reasoning}",
    ]
    if rec.runners_up:
        lines += ["", "### Runners-up"]
        for alt in rec.runners_up:
            lines.append(
                f"- **{alt.brand} {alt.model}** — ${alt.price_usd}, "
                f"{alt.vram_gb}GB, {alt.confidence}% — {alt.reasoning}"
            )
    lines += ["", f"**Summary:** {rec.summary}"]
    return "\n".join(lines)


# --- CHAT LOGIC ---
# This is the original respond() from gradio_app.py, kept verbatim through
# the prompt step. The only change is the final Gemini call: instead of
# returning prose via response.text, we ask for the Recommendation schema
# back and render the validated object. A 3-line override at the end
# replaces the LLM's confidence with the actual retrieval score.
def respond(message, history, max_price, min_vram, max_power, selected_brands):
    # 1. Update query to include 'distances'
    where_filter = {
        "$and": [
            {"price": {"$lte": max_price}},
            {"vram": {"$gte": min_vram}},
            {"power": {"$lte": max_power}},
            {"brand": {"$in": selected_brands}}
        ]
    }

    # Bake slider preferences into the embedding query so retrieval
    # similarity reflects preference-fit, not just topical match.
    enriched_query = (
        f"{message}\n"
        f"Preferences: budget around ${max_price}, "
        f"at least {min_vram}GB VRAM, "
        f"power rating at most {max_power}, "
        f"brands: {', '.join(selected_brands)}."
    )
    results = collection.query(
        query_texts=[enriched_query],
        # n_results=5,
        where=where_filter,
        include=["documents", "metadatas", "distances"]
    )

    # 2. Process results and calculate confidence
    scores_by_model: dict[str, float] = {}
    gpu_list = []
    if results['documents'][0]:
        for idx in range(len(results['documents'][0])):
            doc = results['documents'][0][idx]
            meta = results['metadatas'][0][idx]
            dist = results['distances'][0][idx]
            confidence = round((1 - (dist / 2)) * 100, 1)
            scores_by_model[meta['model']] = confidence
            gpu_list.append(f"Card: {doc} | Confidence Score: {confidence}%")

    context = "\n".join(gpu_list) if gpu_list else "No GPUs match these filters."

    # 3. Update the Prompt for RE-RANKING
    prompt = (
        f"You are a Senior GPU Consultant. The user has these hard filters:\n"
        f"- Max Price: ${max_price}, Min VRAM: {min_vram}GB, Max Power: {max_power}\n\n"
        f"User's question: {message}\n\n"
        f"I have found these potential matches with retrieval confidence scores:\n"
        f"{context}\n\n"
        f"USER TASK:\n"
        f"1. Analyze the matching GPUs.\n"
        f"2. RE-RANK them to best answer the user's specific question above. "
        f"The slider values are hard filters (already applied), not ranking preferences. "
        f"Interpret the question on its own terms — e.g., 'best value' means cheapest "
        f"per unit of the thing they want; 'best card' means highest spec at budget; "
        f"'good for gaming' means gaming-optimized; etc.\n"
        f"3. Explain WHY the top choice is #1 in the context of the user's question.\n"
        f"4. Display the confidence score for each recommendation.\n\n"
        f"Optional tool: get_live_price(model) is available if you need to "
        f"verify a current price, but the prices in the data above are reliable "
        f"for ranking — skip the tool unless verification is genuinely needed."
    )

    # 4. Single call — adds response_schema=Recommendation AND tools=[get_live_price]
    # The model can call the tool mid-decision; final output is still the typed schema.
    top_card_md = ""
    runners_rows: list[list] = []
    try:
        response = gemini_client.models.generate_content(
            model=MODEL,
            contents=prompt,
            config=types.GenerateContentConfig(
                tools=[get_live_price],
                response_mime_type="application/json",
                response_schema=Recommendation,
            ),
        )
        rec: Recommendation | None = response.parsed
        if rec is None:
            reply = f"_(Could not parse structured response)_\n\n{response.text}"
        else:
            # Override LLM-supplied confidence with the actual Chroma score.
            # Order is left as the LLM returned it — its "Best Fit" reasoning
            # accounts for newness/VRAM/budget in ways pure similarity can't.
            for pick in [rec.top_pick, *rec.runners_up]:
                if pick.model in scores_by_model:
                    pick.confidence = scores_by_model[pick.model]
            reply = format_recommendation(rec)
            top_card_md = format_top_card(rec.top_pick)
            runners_rows = format_runners_rows(rec.runners_up)
    except ValidationError as e:
        reply = f"_(Schema validation failed)_\n\n```\n{e}\n```"

    history.append({"role": "user", "content": message})
    history.append({"role": "assistant", "content": reply})
    return "", history, top_card_md, runners_rows

# --- CUSTOM CSS FOR SIZING ---
# This forces the chatbot window and inputs to be significantly longer/taller
custom_css = """
#gpu_chat_log { height: 800px !important; }
#gpu_user_input textarea { min-height: 200px !important; }
"""

# Bind Enter (without Shift) in the message textarea to a click on Send.
# Gradio's multi-line Textbox doesn't fire .submit on Enter, so we listen
# at the document level in the capture phase — this survives any re-render
# Gradio does and runs before its own keydown handlers. Shift+Enter still
# inserts a newline.
enter_to_submit_js = """
(() => {
    document.addEventListener('keydown', (e) => {
        if (e.key !== 'Enter' || e.shiftKey || e.ctrlKey || e.metaKey || e.altKey) return;
        const target = e.target;
        if (!target || target.tagName !== 'TEXTAREA') return;
        if (!target.closest || !target.closest('#gpu_user_input')) return;
        e.preventDefault();
        e.stopPropagation();
        let btn = document.querySelector('#send_btn');
        if (btn && btn.tagName !== 'BUTTON') {
            btn = btn.querySelector('button') || btn;
        }
        if (btn) btn.click();
    }, true);
})();
"""

# --- CUSTOM UI LAYOUT ---
# Pass the custom CSS into the Blocks constructor
with gr.Blocks() as demo:
    gr.Markdown("# 🎮 Advanced GPU Finder")

    with gr.Row():
        # Left Sidebar for Filters (Adjust scale to make it wider/longer)
        with gr.Column(scale=3):
            gr.Markdown("### 🛠 Filters")
            brand_filter = gr.CheckboxGroup(
                choices=["AMD", "Nvidia"],
                value=["AMD", "Nvidia"],
                label="Select Brands"
            )
            price_slider = gr.Slider(100, 1000, value=600, label="Max Price ($)")
            vram_slider = gr.Slider(4, 24, value=8, step=4, label="Min VRAM (GB)")
            power_slider = gr.Slider(1, 10, value=9, label="Max Power Consumption")
            clear_btn = gr.Button("Clear History")

        # Right Side for Chat (Adjust scale to match)
        with gr.Column(scale=6):
            # Added elem_id so our CSS can target this specific chatbot window
            chatbot = gr.Chatbot(elem_id="gpu_chat_log")

            # Added elem_id so our CSS can target the input box height
            msg = gr.Textbox(
                label="Ask a question (e.g., 'What is the best card for me?')",
                elem_id="gpu_user_input",
                lines=3 # This natively makes the box taller right away
            )
            submit_btn = gr.Button("Send", variant="primary", elem_id="send_btn")

    # Structured-view panel — populated from the validated Pydantic object.
    # Card and Dataframe are typed components; they need int/float fields,
    # which is exactly what Pydantic guarantees the LLM produces.
    with gr.Row():
        with gr.Column():
            gr.Markdown("### Structured View (powered by Pydantic)")
            top_pick_card = gr.Markdown(
                value="_Top pick will appear here after your first query._"
            )
            gr.Markdown("#### Runners-up")
            runners_table = gr.Dataframe(
                headers=["GPU", "Price ($)", "VRAM (GB)", "Confidence (%)", "Reasoning"],
                datatype=["str", "number", "number", "number", "str"],
                value=[],
                wrap=True,
            )

    # Connect the components — both Enter (via JS) and the Send button
    # trigger respond(). Outputs now also drive the structured view.
    submit_btn.click(
        respond,
        [msg, chatbot, price_slider, vram_slider, power_slider, brand_filter],
        [msg, chatbot, top_pick_card, runners_table],
    )
    msg.submit(
        respond,
        [msg, chatbot, price_slider, vram_slider, power_slider, brand_filter],
        [msg, chatbot, top_pick_card, runners_table],
    )
    clear_btn.click(lambda: None, None, chatbot, queue=False)

if __name__ == "__main__":
    # Gradio 6.0 theme placement
    demo.launch(theme=gr.themes.Soft(), css=custom_css, js=enter_to_submit_js)
