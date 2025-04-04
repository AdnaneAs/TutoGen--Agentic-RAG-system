from typing import List, Dict, Any, Optional
import pandas as pd
import numpy as np
import requests
from llama_index.core.base.embeddings.base import BaseEmbedding
from llama_index.embeddings.ollama import OllamaEmbedding

def get_available_table_embedding_models() -> List[str]:
    try:
        response = requests.get("http://localhost:11434/api/tags")
        if response.status_code == 200:
            models = response.json().get("models", [])
            table_models = [
                model["name"] for model in models 
                if "embed" in model["name"].lower() or 
                "nomic" in model["name"].lower() or
                "all-minilm" in model["name"].lower() or
                "e5" in model["name"].lower()
            ]
            return table_models
        else:
            return ["nomic-embed-text"]
    except Exception as e:
        print(f"Error fetching table embedding models: {e}")
        return ["nomic-embed-text"]

def get_table_embedding_model(model_name: str):
    return OllamaEmbedding(model_name=model_name)

class TableEmbedding(BaseEmbedding):
    def __init__(self, model_name: str, **kwargs):
        self.model_name = model_name
        self.text_embed_model = OllamaEmbedding(model_name=model_name)
        super().__init__(**kwargs)
    
    def _process_table_text(self, table_text: str) -> str:
        lines = table_text.strip().split('\n')
        if len(lines) > 1 and ',' in lines[0]:
            try:
                df = pd.read_csv(pd.StringIO(table_text))
                structured_text = "Table with columns: " + ", ".join(df.columns) + "\n\n"
                sample_size = min(5, len(df))
                structured_text += "Sample data:\n"
                for i in range(sample_size):
                    row = df.iloc[i]
                    row_text = " | ".join([f"{col}: {row[col]}" for col in df.columns])
                    structured_text += row_text + "\n"
                structured_text += "\nSummary statistics:\n"
                for col in df.select_dtypes(include=[np.number]).columns:
                    structured_text += f"{col}: min={df[col].min()}, max={df[col].max()}, mean={df[col].mean()}\n"
                return structured_text
            except:
                return table_text
        return table_text
    
    def _get_text_embedding(self, text: str) -> List[float]:
        processed_text = self._process_table_text(text)
        return self.text_embed_model._get_text_embedding(processed_text)
    
    def _get_text_embeddings(self, texts: List[str]) -> List[List[float]]:
        processed_texts = [self._process_table_text(text) for text in texts]
        return self.text_embed_model._get_text_embeddings(processed_texts)
    
    def _get_query_embedding(self, query: str) -> List[float]:
        return self.text_embed_model._get_query_embedding(query)