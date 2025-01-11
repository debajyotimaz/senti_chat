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

# Function to detect sentiment and assign an emoji using the LLM
def detect_emotion_llm(message):
    emotion_prompt = f"The user has sent the following message: \"{message}\". Based on this, classify the emotion as one of the following: happy, sad, angry, love, or neutral."
    emotion_response = chain.invoke({"history": "", "question": emotion_prompt})
    
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
def respond(message, chat_history):
    # Format chat history into a string for the model
    history_str = "\n".join(
        f"User: {entry['content']}" if entry["role"] == "user" else f"Assistant: {entry['content']}"
        for entry in chat_history
    )
    
    # Detect emotion and get emoji using LLM
    emoji = detect_emotion_llm(message)
    
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
