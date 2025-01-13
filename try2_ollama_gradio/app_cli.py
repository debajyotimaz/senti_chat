import gradio as gr
import time
from ollama import chat
from ollama import ChatResponse

# Initialize an empty list to keep track of messages
messages = []

# Function to run the Ollama model using the ollama library with chat history
def run_ollama_llm_with_history(user_input):
    global messages
    
    # Use the ollama chat function to get the response
    response = chat(
        'llama3.2:1b',
        messages=messages
        + [
            {'role': 'user', 'content': user_input},
        ],
    )
    
    # Append the message to the chat history
    messages += [
        {'role': 'user', 'content': user_input},
        {'role': 'assistant', 'content': response.message.content},
    ]
    
    # Return the content of the response
    return response.message.content.strip()

# Function to detect sentiment and assign an emoji using the LLM
def detect_emotion_llm(message):
    emotion_prompt = f"The user has sent the following message: \"{message}\". Based on this, classify the emotion as one of the following: happy, sad, angry, love, or neutral."
    emotion_response = chat(
        'llama3.2:1b',
        messages=[
            {'role': 'user', 'content': emotion_prompt},
        ],
    ).message.content
    
    # Map the response to emojis
    if "happy" in emotion_response:
        return "😄"
    elif "sad" in emotion_response:
        return "😢"
    elif "angry" in emotion_response:
        return "😡"
    elif "love" in emotion_response:
        return "😍"
    else:
        return "😐"

# Updated respond function to use the new LLM-based emotion detector
def respond(message, chat_history, image):
    # Format chat history into a string for the model
    history_str = "\n".join(
        f"User: {entry['content']}" if entry["role"] == "user" else f"Assistant: {entry['content']}"
        for entry in chat_history
    )
    
    # Detect emotion and get emoji using LLM
    emoji = detect_emotion_llm(message)
    
    # Generate response using the Ollama model with chat history
    response = run_ollama_llm_with_history(message)    
    
    # Append the user message and assistant response to the history
    chat_history.append({"role": "user", "content": f"{message}<span class='emoji'>{emoji}</span>"})
    chat_history.append({"role": "assistant", "content": response})
    
    # If an image is uploaded, include it in the response
    if image:
        chat_history.append({"role": "user", "content": f"User uploaded an image: {image}"})
    
    # time.sleep(2)
    return "", chat_history

# Define the Gradio interface
custom_css = """
    .emoji {
        font-size: 1.5em;
        position: absolute;
        left: -22px;
        top: 10px;
    }
    .message.user .emoji {
        top: 100%;
        transform: translateY(-50%);
    }
    h1 {
        text-align: center;
    }
"""

with gr.Blocks(css=custom_css) as demo:
    gr.Markdown("<h1>Sentremo</h1>")
    chatbot = gr.Chatbot(type="messages")
    msg = gr.Textbox(placeholder="Type your message here...")
    image = gr.Image(type="pil", label="Upload Image", visible=True)  # Add image upload component
    clear = gr.ClearButton([msg, chatbot, image])

    msg.submit(respond, [msg, chatbot, image], [msg, chatbot])
    image.submit(respond, [msg, chatbot, image], [msg, chatbot])

if __name__ == "__main__":
    demo.launch()
