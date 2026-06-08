import sys
import importlib.util


HAS_VERTEXAI = importlib.util.find_spec("langchain_google_vertexai") is not None


def _patch_ragas_vertexai():
    if "ragas" not in sys.modules:
        return
    if "langchain_community.chat_models.vertexai" not in sys.modules:
        return
    try:
        from langchain_community.chat_models.vertexai import ChatVertexAI
    except ImportError:
        import types
        import langchain_community.chat_models
        vertex_mod = types.ModuleType("langchain_community.chat_models.vertexai")
        if HAS_VERTEXAI:
            from langchain_google_vertexai import ChatVertexAI
            vertex_mod.ChatVertexAI = ChatVertexAI
        else:
            class _DummyChatVertexAI:
                pass
            vertex_mod.ChatVertexAI = _DummyChatVertexAI
        sys.modules["langchain_community.chat_models.vertexai"] = vertex_mod
        langchain_community.chat_models.vertexai = vertex_mod


def patch_ragas():
    _patch_ragas_vertexai()
