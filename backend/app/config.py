from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    openai_api_key: str = ""
    openai_base_url: str = "http://localhost:20128/v1"
    llm_model: str = "ag/gemini-3.5-flash-low"
    llm_model_fallback: str = "kr/deepseek-3.2"
    llm_model_legacy: str = "cx/gpt-5.5"
    hf_token_1: str = ""
    hf_token_2: str = ""
    qdrant_url: str = "http://localhost:6333"
    qdrant_collection: str = "calculus_book"
    embedding_model: str = "BAAI/bge-small-en-v1.5"
    reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    chunk_size: int = 1024
    chunk_overlap: int = 128
    top_k_retrieve: int = 30
    top_k_rerank: int = 5
    top_k_mmr: int = 5
    mmr_lambda: float = 0.7
    data_dir: str = "data"
    request_delay: float = 3.0
    max_retries: int = 3
    use_hyde: bool = False
    use_self_consistency: bool = True
    self_consistency_samples: int = 3
    self_consistency_temperature: float = 0.3
    self_consistency_use_semantic: bool = True
    use_reranker: bool = False
    reranker_method: str = "hybrid"  # token_overlap, bm25, cross_encoder, hybrid
    rerank_skip_threshold: int = 0
    use_compression: bool = True
    compression_mode: str = "auto"
    compression_skip_threshold: int = 2000
    relevance_threshold: float = 0.15
    response_cache_enabled: bool = True
    response_cache_max_size: int = 100

    # Confidence/UQ settings
    confidence_abstain_threshold: float = 0.4
    confidence_warn_threshold: float = 0.6

    # NLI verification settings
    use_nli_verifier: bool = True
    nli_model_name: str = "roberta-large-mnli"
    nli_verification_weight: float = 0.5
    nli_cache_max_size: int = 1000
    nli_device: str = "cpu"

    model_config = {"env_file": ".env", "env_file_encoding": "utf-8", "extra": "ignore"}

    def get_hf_token(self) -> str:
        for token in [self.hf_token_1, self.hf_token_2]:
            if token:
                return token
        return ""


settings = Settings()
