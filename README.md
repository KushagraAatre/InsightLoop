# 🌐 InsightLoop – Multimodal Review Analytics & Persona Studio

## 🚀 What is InsightLoop?

InsightLoop is an advanced, AI-powered, multimodal review analytics and market research platform. It enables brands to analyze customer feedback from text, images, and audio, extract actionable insights, generate dynamic user personas, and simulate new product launches—all in a single, beautiful Streamlit app.

## ✨ Key Features

- **Multimodal Review Processing**: Analyze text, image (OCR), and audio reviews
- **Sentiment & Emotion Analysis**: See pie charts, trend lines, and histograms using HuggingFace models
- **Theme/Keyword Extraction**: Find top themes and trends from review data
- **Persona Generation**: Auto-generate 4 rich customer personas using LLMs (Groq/LLAMA 3)
- **Persona Analytics**: View persona-wise review distribution, sentiment, emotion, and highlights
- **Product Launch Simulator**: Enter new product ideas and get persona-specific feedback, risks, and marketing strategies
- **Modern UI**: Neon-themed, interactive, and responsive design powered by Plotly and Streamlit

## 🗂️ Project Structure

```
.

├── streamlit_app.py        # Main app entry point (sometimes main.py)
├── pages/
│   ├── prt111.py               # Main review analytics dashboard
│   ├── persona.py              # Persona analysis & visualization
│   └── newprod.py              # Product launch studio
├── review_files/               # Directory for raw review files (txt/png/wav)
├── data.csv                    # Raw review metadata
└── personas.json               # (temp) Persona definitions (overwritten at runtime)
├── .env                        # Your secrets (API keys, paths) — DO NOT COMMIT
├── requirements.txt            # All dependencies
├── README.md                   # This file
└── ...
```

## 📦 Requirements

- Python 3.8+
- See `requirements.txt` for all Python dependencies (streamlit, transformers, plotly, groq, pytesseract, scikit-learn, etc.)
- Groq API Key (for LLM features)
- (Optional) HuggingFace Transformers cache for models

## ⚙️ Setup Instructions

### 1. Clone the repository

```bash
git clone https://github.com/yourusername/InsightLoop.git
cd InsightLoop
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Prepare Your .env File

Create a `.env` file at the project root (same folder as README.md):

```ini
GROQ_API_KEY=sk-xxxx-your-groq-api-key
PERSONA_PATH=/tmp/personas.json
TRANSFORMERS_CACHE=/tmp/hf_cache
HF_HOME=/tmp/huggingface
HF_DATASETS_CACHE=/tmp/huggingface
```

> **⚠️ Important**: Do NOT commit `.env` to GitHub. The `/tmp/` paths are used for cloud deployment compatibility (Streamlit Cloud, HF Spaces, etc.).

### 4. File Storage for Reviews

- Place your review `.txt`, `.png` (images), and `.wav` (audio) files inside `src/review_files/`
- Make sure your CSV (`data.csv`) contains a `review_file` column referencing these files

### 5. First Run

```bash
streamlit run src/streamlit_app.py
```

or (if using main.py as entry):

```bash
streamlit run src/main.py
```

## 🖥️ How it Works

### A. Review Analytics (prt111.py)
- Loads and preprocesses review data (extracts text from images/audio as needed)
- Performs sentiment and emotion analysis (using HuggingFace models)
- Visualizes insights with interactive charts (Plotly)
- Summarizes findings using Groq LLM for punchy insights

### B. Persona Analysis (persona.py)
- Uses Groq LLM to generate 4 unique personas from the reviews
- Assigns each review to a persona using TF-IDF similarity
- Displays persona-specific statistics, highlights, and actionable recommendations

### C. New Product Launch Studio (newprod.py)
- Enter a product concept/description
- The app simulates reactions, alignments, risks, and marketing strategies for each persona
- Projects market share and sentiment for your new product idea
- Summarizes all results with LLM-powered executive summaries

## ☁️ Deployment Notes (for Streamlit Cloud / HuggingFace Spaces)

- All files written at runtime (e.g., `personas.json`, any temp CSVs, model downloads) must be written to `/tmp/`
- All cache and model paths are set via environment variables to `/tmp/`
- Add `.env` and any sensitive files to `.gitignore`

### Sample .gitignore

```
.env
src/personas.json
*.pyc
__pycache__/
review_files/
*.log
```

## ❗ Troubleshooting

| Error | Cause/Fix |
|-------|-----------|
| `PermissionError: [Errno 13] Permission denied` | You're writing to a non-writable path. Change output paths to `/tmp/` |
| `No personas found. Please generate personas first.` | Run the Persona Analysis page first to generate personas |
| `KeyError: 'GROQ_API_KEY'` | Ensure you set `GROQ_API_KEY` in your `.env` file |
| HuggingFace model download errors (`.cache` permissions) | Set cache env variables to `/tmp/` as described above |
| Charts not displaying | Check for missing dependencies in `requirements.txt` |

## 🧠 Credits & Acknowledgements

- **LLM Summarization/Persona**: Groq / Meta Llama 3
- **Sentiment/Emotion Analysis**: HuggingFace Transformers
- **Data Visualization**: Plotly, Streamlit
- **OCR & Speech**: pytesseract, SpeechRecognition
- **UI Design**: Neon inspiration from HyperUI and Glassmorphism

## 🙏 Contributions

Pull requests and feature requests are welcome! Please open an issue if you find bugs or have ideas for improvement.


## 🚩 Contact

**Project by Kushagra Aatre**  
Feel free to reach out for questions, collaboration, or feedback!

---

**Happy Insight Mining! 🚀**
