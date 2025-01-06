from flask import Flask, request, render_template
import torch
from transformers import pipeline
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

# Initialize Flask app
app = Flask(__name__)

# Load the LLaMA model (LLaMA 2 example)
model_id = "meta-llama/Llama-3.2-1B"
cache_dir = "/data1/debajyoti/test/llms/models/llma3_8b_instruct"  # Replace with a valid model name or local path
# Load model and tokenizer from the cache directory
tokenizer = AutoTokenizer.from_pretrained(model_id, cache_dir=cache_dir)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    cache_dir=cache_dir
)

# Initialize the text-generation pipeline
chat_pipeline = pipeline("text-generation", model=model, tokenizer=tokenizer, model_kwargs={"torch_dtype": torch.bfloat16}, device_map="auto")

# Chat history to maintain context
chat_history = []

# Function to generate chatbot response
def get_response(user_input):
    global chat_history

    # Prepare the conversation prompt
    conversation_prompt = "\n".join(
        [
            f"User: {entry['content']}" if entry["role"] == "user" else f"Assistant: {entry['content']}"
            for entry in chat_history
        ]
    )
    prompt = f"{conversation_prompt}\nUser: {user_input}\nAssistant:"

    # Generate a response using the text-generation pipeline
    response = chat_pipeline(prompt, max_length=512, num_return_sequences=1, temperature=0.7)
    reply = response[0]["generated_text"].split("Assistant:")[-1].strip()

    # Append the user input and bot response to chat history
    chat_history.append({"role": "user", "content": user_input})
    chat_history.append({"role": "assistant", "content": reply})

    return reply

# Define the homepage route
@app.route("/", methods=["GET", "POST"])
def home():
    if request.method == "POST":
        # Get user input from the form
        user_input = request.form.get("text")
        if not user_input:
            result = {"error": "Please enter a message"}
        else:
            # Generate chatbot response
            bot_response = get_response(user_input)
            result = {"user_input": user_input, "bot_response": bot_response}

        return render_template("chat.html", result=result, history=chat_history)

    return render_template("chat.html", result=None, history=chat_history)

# Run the Flask app
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)

