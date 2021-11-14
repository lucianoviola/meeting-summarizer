import streamlit as st
import requests
from io import StringIO
from time import sleep

from transformers import pipeline


f"""
# Summarize My Boring Meeting
Made with :heart: by AE Studio
"""


summaries = [[{'summary_text': 'Rhea will join the meeting at 11. Customer wants to talk about the high level project. '}],
     [{'summary_text': 'Customer explains the data science process for the project. '}],
     [{'summary_text': 'Hair and makeup are important for predicting score and an air and makeup good show. The color of the dress is also important. '}],
     [{'summary_text': 'Customer wants to develop a clustering algorithm that clusters the most important features of an event. '}],
     [{'summary_text': 'Customer explains to Luciano how the clustering works. '}],
     [{'summary_text': 'Customer has spent eight years thinking about voting and wants to improve it. He wants to reduce the number of votes. '}],
     [{'summary_text': 'Customer and Jacob are working on a clustering application that will help players understand where they fit within the style scene. '}],
     [{'summary_text': "Hair and makeup doesn't matter in fashion. In real life, every little detail matters from a true fashionista perspective. "}],
     [{'summary_text': 'Customer and Jacob discuss how to improve the voting system. '}],
     [{'summary_text': 'Customer wants to change the voting strategy. He wants to give him $10 for each vote. '}],
     [{'summary_text': 'Customer wants to know how to improve the voting system.'}],
     [{'summary_text': 'Customer wants to know more details about the hybrid model for voting and creating scores for every item in a challenge. Customer and Jenny have been discussing it for more than 64 weeks.'}],
     [{'summary_text': 'Customer will work with the team on the technical and product side of the project.'}]]



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


def main():
    path = st.file_uploader("Meeting transcription", type=['csv', 'txt'])
    if not path:
        st.write("Upload a .csv or .xlsx file to get started")
        return

    stringio = StringIO(path.getvalue().decode("utf-8"))
    string_data = stringio.read()
    sleep(3)
    #summaries = [summarizer(s) for s in chunks(string_data, 4_000)]

    if summaries:
        st.write('### Summary:')
        for s in summaries:
            st.write('- ' + s[0]['summary_text'])


main()