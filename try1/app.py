from flask import Flask, request, render_template
from transformers import BertTokenizer, BertForSequenceClassification
import torch

# Initialize Flask app
app = Flask(__name__)

# Load pre-trained BERT model and tokenizer
MODEL_NAME = "bert-base-uncased"  # Replace with your fine-tuned model path if applicable
tokenizer = BertTokenizer.from_pretrained(MODEL_NAME)
model = BertForSequenceClassification.from_pretrained(MODEL_NAME)
model.eval()  # Set the model to evaluation mode

# Define a function for model inference
def predict(text):
    # Tokenize the input text
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
    
    # Perform inference
    with torch.no_grad():
        outputs = model(**inputs)
    
    # Get the predicted label (assuming binary classification)
    logits = outputs.logits
    probabilities = torch.softmax(logits, dim=-1).squeeze()
    predicted_class = torch.argmax(probabilities).item()
    confidence = probabilities[predicted_class].item()
    
    return {"predicted_class": predicted_class, "confidence": confidence}

# Define the homepage route
@app.route("/", methods=["GET", "POST"])
def home():
    if request.method == "POST":
        # Get text input from the form
        text = request.form.get("text")
        if not text:
            result = {"error": "Please enter some text"}
        else:
            # Perform prediction
            result = predict(text)
        
        # Render the result on the page
        return render_template("index.html", text=text, result=result)
    
    # Render the page for GET request
    return render_template("index.html", text="", result=None)

# Run the Flask app
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)

