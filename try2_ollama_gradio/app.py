import gradio as gr
import time
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama.llms import OllamaLLM

# Define the Ollama model and prompt template
template = """The following is a conversation between a user and an assistant:

{history}

User: {question}

Assistant: Let's think step by step."""
prompt = ChatPromptTemplate.from_template(template)
model = OllamaLLM(model="llama3.2:1b")
chain = prompt | model

# Function to detect sentiment and assign an emoji
def detect_emotion(message):
    if any(word in message.lower() for word in ["sad", "stressed", "down", "unhappy", "depressed"]):
        return "😢"
    elif any(word in message.lower() for word in ["happy", "excited", "great", "fantastic", "joy"]):
        return "😄"
    elif any(word in message.lower() for word in ["angry", "mad", "frustrated", "furious"]):
        return "😡"
    elif any(word in message.lower() for word in ["love", "like", "enjoy", "adore"]):
        return "😍"
    else:
        return "😐"

# Function to generate responses using the Ollama model
def respond(message, chat_history):
    # Format chat history into a string for the model
    history_str = "\n".join(
        f"User: {entry['content']}" if entry["role"] == "user" else f"Assistant: {entry['content']}"
        for entry in chat_history
    )
    
    # Detect emotion and get emoji
    emoji = detect_emotion(message)
    
    # Generate response using the Ollama model
    response = chain.invoke({"history": history_str, "question": message})  # Add chat history and current message
    
    # Append the user message and assistant response to the history
    chat_history.append({"role": "user", "content": f"{message}<span class='emoji'>{emoji}</span>"})
    chat_history.append({"role": "assistant", "content": response})
    
    time.sleep(2)
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
"""

with gr.Blocks(css=custom_css) as demo:
    gr.Markdown("<h1>SentiChat</h1>")
    chatbot = gr.Chatbot(type="messages")
    msg = gr.Textbox(placeholder="Type your message here...")
    clear = gr.ClearButton([msg, chatbot])

    msg.submit(respond, [msg, chatbot], [msg, chatbot])

if __name__ == "__main__":
    demo.launch()
