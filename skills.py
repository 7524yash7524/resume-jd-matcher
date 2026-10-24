"""A simple list of data / ML skills and a function to find them in text.

The list is hand-made on purpose: it is easy to read, easy to extend,
and gives results you can explain in an interview.
"""

import re

# skill name shown to the user -> other ways people write it
SKILLS = {
    # languages
    "Python": ["python"],
    "SQL": ["sql"],
    "R": [r"\bR\b"],
    "C++": [r"c\+\+"],
    "Java": [r"\bjava\b"],
    # data libraries
    "Pandas": ["pandas"],
    "NumPy": ["numpy"],
    "Scikit-learn": ["scikit-learn", "sklearn", "scikit learn"],
    "Statsmodels": ["statsmodels"],
    "Matplotlib": ["matplotlib"],
    "Seaborn": ["seaborn"],
    "Plotly": ["plotly"],
    # deep learning
    "Deep Learning": ["deep learning", r"\bDL\b"],
    "PyTorch": ["pytorch", "torch"],
    "TensorFlow": ["tensorflow"],
    "Keras": ["keras"],
    "Transformers": ["transformer", "transformers", "hugging face", "huggingface"],
    "NLP": [r"\bnlp\b", "natural language processing"],
    "Computer Vision": ["computer vision", r"\bcv model", "image classification"],
    # gen ai
    "Generative AI": ["generative ai", "gen ai", "genai"],
    "LLMs": [r"\bllms?\b", "large language model"],
    "LangChain": ["langchain"],
    "LlamaIndex": ["llamaindex", "llama index"],
    "RAG": [r"\brag\b", "retrieval augmented", "retrieval-augmented"],
    "Prompt Engineering": ["prompt engineering", "prompting"],
    "Vector Databases": ["vector database", "vector store", "faiss", "chroma", "pinecone"],
    # ml concepts
    "Machine Learning": ["machine learning", r"\bml\b"],
    "Time Series": ["time series", "time-series", "forecasting", "arima"],
    "Statistics": ["statistics", "statistical"],
    "Feature Engineering": ["feature engineering"],
    "EDA": [r"\beda\b", "exploratory data analysis"],
    "Data Cleaning": ["data cleaning", "cleaning"],
    "Model Deployment": ["deploy", "deployed", "deployment", "productionize", "production"],
    # big data / tools
    "PySpark": ["pyspark", r"\bspark\b"],
    "Hadoop": ["hadoop"],
    "Git": [r"\bgit\b", "github"],
    "Streamlit": ["streamlit"],
    "Jupyter": ["jupyter"],
    "Docker": ["docker"],
    "AWS": [r"\baws\b", "amazon web services"],
    "Azure": ["azure"],
    "GCP": [r"\bgcp\b", "google cloud"],
    "Power BI": ["power bi", "powerbi"],
    "Tableau": ["tableau"],
    "Excel": ["excel"],
}


def _pattern(alias: str) -> re.Pattern:
    """Turn an alias into a regex. Plain words get word boundaries."""
    if "\\" in alias:  # already a regex
        return re.compile(alias, re.IGNORECASE)
    return re.compile(r"(?<![a-z0-9])" + re.escape(alias) + r"(?![a-z0-9])", re.IGNORECASE)


_COMPILED = {name: [_pattern(a) for a in aliases] for name, aliases in SKILLS.items()}


def find_skills(text: str) -> set[str]:
    """Return the set of skills from SKILLS that appear in the text."""
    found = set()
    for name, patterns in _COMPILED.items():
        if any(p.search(text) for p in patterns):
            found.add(name)
    return found
