from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    openai_api_key: str = ""
    openai_base_url: str = "http://localhost:20128/v1"
    llm_model: str = "cx/gpt-5.5"
    hf_token_1: str = ""
    hf_token_2: str = ""
    qdrant_url: str = "http://localhost:6333"
    qdrant_collection: str = "calculus_book"
    embedding_model: str = "all-MiniLM-L6-v2"
    reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    chunk_size: int = 512
    chunk_overlap: int = 50
    top_k_retrieve: int = 30
    top_k_rerank: int = 5
    data_dir: str = "data"

    model_config = {"env_file": ".env", "env_file_encoding": "utf-8"}

    def get_hf_token(self) -> str:
        for token in [self.hf_token_1, self.hf_token_2]:
            if token:
                return token
        return ""


settings = Settings()
