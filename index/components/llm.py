import os
from llama_index.llms import OpenAI  # Change this line
from llama_index.embeddings import OpenAIEmbedding
from llama_index import set_global_service_context, ServiceContext
from components.log import log

llm_engine = None


def init_llm():
    api_key = os.getenv("OPENAI_API_KEY")  
    api_base = "https://dev-hub.agentartificial.com"  
    llm_model = os.getenv("OPENAI_LLM_MODEL")  
    embed_model = os.getenv("OPENAI_EMBED_MODEL")  

    llm_engine = OpenAI(  
        model=llm_model,
        api_key=api_key,
        api_base=api_base,
    )

    embed_engine = OpenAIEmbedding(
        model=embed_model,
        deployment_name=embed_model,
        api_key=api_key,
        api_base="https://api.openai.com",  # Changed this line to point to regular OpenAI
    )

    service_context = ServiceContext.from_defaults(
        llm=llm_engine,
        embed_model=embed_engine,
    )

    set_global_service_context(service_context)

    log.info("init llm ok")
