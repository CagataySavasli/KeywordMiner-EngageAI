from sklearn.feature_extraction.text import TfidfVectorizer
from gensim.models import Word2Vec
from nltk.corpus import stopwords
from transformers import AutoTokenizer, AutoModel
import torch
from collections import defaultdict
import nltk
import re

nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

def keyword_extractor_factory(*args):
    if args[0] == "TF-IDF":
        return extract_keywords_tfidf(args[1], args[2])
    elif args[0] == "Word2Vec":
        return extract_keywords_word2vec(args[1], args[2])
    elif args[0] == "BERT":
        return extract_keywords_transformer(args[1], args[2], args[3])

def extract_keywords_tfidf(texts, top_n):
    vectorizer = TfidfVectorizer(stop_words='english', max_features=top_n)
    tfidf_matrix = vectorizer.fit_transform(texts)
    keywords = vectorizer.get_feature_names_out()

    return keywords

def extract_keywords_word2vec(texts, top_n):
    tokenized_texts = [
        [word for word in re.findall(r'\b\w+\b', text.lower()) if word not in stop_words]
        for text in texts
    ]


    model = Word2Vec(sentences=tokenized_texts, vector_size=100, window=5, min_count=1, workers=4)
    keywords = model.wv.index_to_key[:top_n]
    return keywords

def extract_keywords_transformer(texts, top_n, chunk, model_name="bert-base-uncased"):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name, output_attentions=True)

    token_score_sum = defaultdict(float)
    token_count = defaultdict(int)

    for text in texts:
        inputs = tokenizer(text, return_tensors="pt", truncation=False)
        input_ids = inputs["input_ids"].squeeze()

        num_segments = (len(input_ids) + chunk - 1) // chunk
        for i in range(num_segments):
            start_idx = i * chunk
            end_idx = start_idx + chunk

            segment_ids = input_ids[start_idx:end_idx]
            segment_inputs = {"input_ids": segment_ids.unsqueeze(0)}

            with torch.no_grad():
                outputs = model(**segment_inputs)

            attentions = outputs.attentions[-1]
            attention_weights = attentions.mean(dim=1).squeeze(0)
            token_importance = attention_weights.sum(dim=0)

            tokens = tokenizer.convert_ids_to_tokens(segment_ids)

            for token, score in zip(tokens, token_importance.tolist()):
                if token.isalpha():  # Sadece kelimeleri sakla
                    token_score_sum[token] += score
                    token_count[token] += 1

    token_avg_scores = {token: token_score_sum[token] / token_count[token] for token in token_score_sum}

    sorted_tokens = sorted(token_avg_scores.items(), key=lambda x: x[1], reverse=True)

    keywords = [word for (word, score) in sorted_tokens if word not in stop_words][:top_n]

    return keywords