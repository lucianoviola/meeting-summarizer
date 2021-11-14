import streamlit as st
import requests
from io import StringIO

from transformers import pipeline


"""
# Boring Meeting Summarizer
"""


def chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i:i + n]


def query(payload, model_id, api_token):
    headers = {"Authorization": f"Bearer {api_token}"}
    API_URL = f"https://api-inference.huggingface.co/models/{model_id}"
    response = requests.post(API_URL, headers=headers, json=payload)
    return response.json()

@st.cache
def load_model():
    return pipeline("summarization", model="lidiya/bart-large-xsum-samsum")


def main():
    summarizer = load_model()
    path = st.file_uploader("Upload transcription", type=['csv', 'txt'])
    if not path:
        st.write("Upload a .csv or .xlsx file to get started")
        return

    stringio = StringIO(path.getvalue().decode("utf-8"))
    string_data = stringio.read()

    summaries = [summarizer(s) for s in chunks(string_data, 4_000)]

    if summaries:
        st.write('### Summary:')
        for s in summaries:
            st.write('- ' + s[0]['summary_text'])


main()