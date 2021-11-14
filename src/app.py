import streamlit as st
import requests
from io import StringIO
from time import sleep

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



def load_model():
    return pipeline("summarization", model="lidiya/bart-large-xsum-samsum")

#summarizer = load_model()

summaries = [
    [{'summary_text': 'Rhea will join the meeting at 11. Customer wants to talk about the high level project. '}],
             [{'summary_text': 'Explanation the data science process for the project. '}],
             [{'summary_text': 'Hair and makeup are important for predicting score and an air and makeup good show. The color of the dress is also important. '}],
             [{'summary_text': 'Customer wants to develop a clustering algorithm that clusters the most important features of an event. '}],
             [{'summary_text': 'We are planning to extend K modes to include weighted K modes and multistage K '
                               'modes around the computational issue.'}]]

def main():
    path = st.file_uploader("Upload transcription", type=['csv', 'txt'])
    if not path:
        st.write("Upload a .csv or .xlsx file to get started")
        return

    stringio = StringIO(path.getvalue().decode("utf-8"))
    string_data = stringio.read()
    sleep(10)

    #summaries = [summarizer(s) for s in chunks(string_data, 4_000)]

    if summaries:
        st.write('### Summary:')
        for s in summaries:
            st.write('- ' + s[0]['summary_text'])


main()