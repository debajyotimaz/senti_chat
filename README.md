# 🫂 SENTREMO Chatbot

Sentremo is an emotion-aware chatbot designed to improve the current emotional condition of the user. It detects the sentiment of user input and generates an appropriate emoji to suggest the sentiment. 

This repository contains different approaches to developing Sentremo using various tools and frameworks.

## ✨ Features

### Core Capabilities

- **🎯 Emotion Detection**
  - Real-time sentiment analysis of user input
  - Context-aware emotional understanding

- **💝 Emotional Support**
  - Personalized empathetic responses
  - Mood improvement strategies
  - Adaptive conversation flow

- **😊 Dynamic Emoji Integration**
  - Context-appropriate emoji suggestions
  - Enhanced emotional expression
    

## 📂 Project Structure

```
sentremo/
├── try1_flask/
│   ├── app.py              # Flask application core
│   ├── templates/          # HTML template files
│   └── README.md          # Flask implementation guide
│
├── try2_ollama_gradio/
│   ├── app_cli.py         # Gradio CLI application
│   └── README.md          # Gradio implementation guide
│
└── README.md              # Main documentation providing an overview of the project and directory structure.
```

## 🚀 Getting Started

### Prerequisites

- Python 3.8+
- pip package manager

### Required Dependencies

```bash
pip install flask gradio torch transformers
```

### Installation

1. **Clone the Repository**
   ```bash
   git clone https://github.com/yourusername/sentremo.git
   cd sentremo
   ```

2. **Set Up Virtual Environment** (Recommended)
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

## 💡 Usage

### Flask Implementation
```bash
cd try1_flask
python app.py
```
Then navigate to `http://localhost:5000` in your browser.

### Gradio Implementation
```bash
cd try2_ollama_gradio
python app_cli.py
```
   
## 🤝 Contributing

We welcome contributions! Here's how you can help:

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

<div align="center">

Made with ❤️ by the Sentremo Team

</div>
