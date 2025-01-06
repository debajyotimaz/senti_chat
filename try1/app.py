from flask import Flask, request, jsonify
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

# Define a route for prediction
@app.route("/predict", methods=["POST"])
def predict_route():
    # Get JSON input from the user
    data = request.json
    if "text" not in data:
        return jsonify({"error": "Missing 'text' in request data"}), 400
    
    text = data["text"]
    
    # Perform prediction
    result = predict(text)
    
    return jsonify(result)

# Define a health check route
@app.route("/", methods=["GET"])
def health_check():
    return jsonify({"message": "BERT model inference API is running!"})

# Run the Flask app
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
