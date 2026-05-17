import gradio as gr
import chromadb
from google import genai
from google.genai import types
import os
from dotenv import load_dotenv

# Load API Key
load_dotenv()

# --- SETUP ---
gemini_client = genai.Client()
chroma_client = chromadb.Client()
collection = chroma_client.get_or_create_collection(name="gpu_db3")

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

# --- CHAT LOGIC ---
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

    results = collection.query(
        query_texts=[message],
        n_results=5,
        where=where_filter,
        include=["documents", "metadatas", "distances"] # Get scores too!
    )
    
    # 2. Process results and calculate confidence
    gpu_list = []
    if results['documents'][0]:
        for idx in range(len(results['documents'][0])):
            doc = results['documents'][0][idx]
            # Convert distance to a 0-100 similarity score
            # (Note: Chroma distance 0 is perfect, 2 is maximum distance)
            dist = results['distances'][0][idx]
            confidence = round((1 - (dist / 2)) * 100, 1) 
            gpu_list.append(f"Card: {doc} | Confidence Score: {confidence}%")
    
    context = "\n".join(gpu_list) if gpu_list else "No GPUs match these filters."
    
    # 3. Update the Prompt for RE-RANKING
    prompt = (
        f"You are a Senior GPU Consultant. The user has these requirements:\n"
        f"- Max Price: ${max_price}, Min VRAM: {min_vram}GB, Max Power: {max_power}\n\n"
        f"I have found these potential matches with retrieval confidence scores:\n"
        f"{context}\n\n"
        f"USER TASK:\n"
        f"1. Analyze the matching GPUs.\n"
        f"2. RE-RANK them by 'Best Fit'. Prioritize cards that offer the most VRAM while staying closest to the user's budget.\n"
        f"3. Explain WHY the top choice is #1.\n"
        f"4. Display the confidence score for each recommendation."
    )
    
    response = gemini_client.models.generate_content(
        model="gemini-3-flash-preview",
        contents=prompt
    )
    
    history.append({"role": "user", "content": message})
    history.append({"role": "assistant", "content": response.text})
    return "", history

# --- CUSTOM CSS FOR SIZING ---
# This forces the chatbot window and inputs to be significantly longer/taller
custom_css = """
#gpu_chat_log { height: 800px !important; } 
#gpu_user_input textarea { min-height: 200px !important; }
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

    # Connect the components
    msg.submit(
        respond, 
        [msg, chatbot, price_slider, vram_slider, power_slider, brand_filter], 
        [msg, chatbot]
    )
    clear_btn.click(lambda: None, None, chatbot, queue=False)

if __name__ == "__main__":
    # Gradio 6.0 theme placement
    demo.launch(theme=gr.themes.Soft(), css=custom_css)