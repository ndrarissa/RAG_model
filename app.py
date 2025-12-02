import streamlit as st
from src.rag import answer_with_context

st.set_page_config(page_title='RAG Corpus-Only Chat')
st.title('RAG Chatbot — Corpus Only')

if 'history' not in st.session_state:
    st.session_state.history = []

query = st.text_input('Ask a question:')
if st.button('Send') and query.strip():
    answer = answer_with_context(query)
    st.session_state.history.append({'q': query, 'a': answer})

for exchange in reversed(st.session_state.history):
    st.markdown(f"**Q:** {exchange['q']}")
    st.markdown(f"**A:** {exchange['a']}")
    st.write('---')
