from flask import Flask, request, render_template
from ollama import chat

app = Flask(__name__)

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
    messages.append({'role': 'user', 'content': user_input})
    messages.append({'role': 'assistant', 'content': response.message.content})
    
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

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        user_input = request.form['text']
        if user_input.strip():
            # Detect emotion and get emoji using LLM
            emoji = detect_emotion_llm(user_input)
            # Generate response using the Ollama model with chat history
            response = run_ollama_llm_with_history(user_input)
            # Append emoji as a reaction to user message
            messages[-2]['emoji'] = emoji
    
    return render_template('index.html', history=messages)

if __name__ == "__main__":
    app.run(debug=True, port=5001)
