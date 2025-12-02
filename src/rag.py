import os
from openai import OpenAI
from src.retriever import Retriever
from dotenv import load_dotenv

load_dotenv()
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
client = OpenAI(api_key=OPENAI_API_KEY) if OPENAI_API_KEY else None

retriever = Retriever()

SYSTEM_INSTRUCTION = (
"You are a helpful assistant. Answer ONLY using the provided CONTEXT. "
"If the answer cannot be found in the context, respond: "
"'Sorry, the information is not available in the corpus.'"
)

def answer_with_context(question, top_k=3):
    context_chunks = retriever.retrieve(question, top_k=top_k)
    context = "\n\n".join(context_chunks)

    prompt = f"CONTEXT:\n{context}\n\nQUESTION:\n{question}\n\nFinal Answer:"

    if client is None:
    # Fallback: simple deterministic rule (very basic)
        return "[No LLM configured] Context used:\n\n" + context

    response = client.responses.create(
        model="gpt-4o-mini", # change to a model you have access to
        input=[
        {"role": "system", "content": SYSTEM_INSTRUCTION},
        {"role": "user", "content": prompt}
        ],
        max_output_tokens=512
        )

    # Adapt response parsing as needed depending on client version
    text = response.output_text if hasattr(response, 'output_text') else response['output'][0]['content'][0]['text']
    return text

