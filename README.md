# KeywordMiner-EngageAI

**KeywordMiner-EngageAI** is a project that extracts keywords from news articles retrieved from Google News using **TF-IDF**, **Word2Vec**, and **BERT**. The dataset includes articles related to keywords such as `US elections`, `Donald Trump`, and `Kamala Harris`.

## Features

- **Data Collection:**
  - Automatically downloads news articles from Google News based on specified keywords.
- **Keyword Extraction:**
  - **TF-IDF:** Identifies significant keywords using Term Frequency-Inverse Document Frequency.
  - **Word2Vec:** Detects semantically important keywords using word embeddings.
  - **BERT:** Leverages transformer-based contextual embeddings to identify context-aware keywords.
- **Modular Design:** Allows for easy integration of additional extraction methods.
- **Poetry for Dependency Management:** All dependencies and the environment are managed using `poetry`.

---

## Installation

Follow these steps to set up and run the project:

### 1. Clone the Repository
```bash
git clone https://github.com/username/KeywordMiner-EngageAI.git
cd KeywordMiner-EngageAI
```

### 2. Install Dependencies
```bash
poetry install
```

### 3. Activate the Poetry Shell
```bash
poetry shell
```
