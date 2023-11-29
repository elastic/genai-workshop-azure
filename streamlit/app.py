import os
import streamlit as st
import openai
from elasticsearch import Elasticsearch
from string import Template
import elasticapm

# Configure OpenAI client
openai.api_key = os.environ['OPENAI_API_KEY']
openai.api_base = os.environ['OPENAI_API_BASE']
openai.default_model = os.environ['OPENAI_API_ENGINE']
openai.verify_ssl_certs = False

# Configure APM and Elasticsearch clients
@st.cache_resource
def initElastic():
    os.environ['ELASTIC_APM_SERVICE_NAME'] = "genai_workshop_lab_2-2"
    apmclient = elasticapm.Client()
    elasticapm.instrument()

    if 'ELASTIC_CLOUD_ID' in os.environ:
      es = Elasticsearch(
        cloud_id=os.environ['ELASTIC_CLOUD_ID'],
        api_key=(os.environ['ELASTIC_APIKEY_ID'], os.environ['ELASTIC_APIKEY_SECRET']),
        request_timeout=30
      )
    else:
      es = Elasticsearch(
        os.environ['ELASTIC_URL'],
        api_key=(os.environ['ELASTIC_APIKEY_ID'], os.environ['ELASTIC_APIKEY_SECRET']),
        request_timeout=30
      )

    if os.environ['ELASTIC_PROXY'] != "True":
        openai.api_type = os.environ['OPENAI_API_TYPE']
        openai.api_version = os.environ['OPENAI_API_VERSION']

    return apmclient, es
apmclient, es = initElastic()

# Set our data index
index = os.environ['ELASTIC_INDEX_DOCS']

# Run an Elasticsearch query using BM25 relevance scoring
@elasticapm.capture_span("bm25_search")
def search_bm25(query_text, es):
    query = {
        "match": {
            "body_content": query_text
        }
    }

    fields= [
        "title",
        "url",
        "position",
        "body_content"
      ]

    collapse= {
      "field": "title.enum"
    }

    resp = es.search(index=index,
                     query=query,
                     fields=fields,
                     collapse=collapse,
                     size=1,
                     source=False)

    body = resp['hits']['hits'][0]['fields']['body_content'][0]
    url = resp['hits']['hits'][0]['fields']['url'][0]

    return body, url

# Run an Elasticsearch query using ELSER relevance scoring
@elasticapm.capture_span("elser_search")
def search_elser(query_text, es):
    query = {
      "text_expansion": {
        "ml.inference.chunk_expanded.tokens": {
          "model_id": ".elser_model_1",
          "model_text": query_text
        }
      }
    }

    fields = [
      "title",
      "url",
      "position",
      "body_content"
    ]

    collapse = {
      "field": "title.enum"
    }

    resp = es.search(index=index,
                     query=query,
                     fields=fields,
                     collapse=collapse,
                     size=1,
                     source=False)

    body = resp['hits']['hits'][0]['fields']['body_content'][0]
    url = resp['hits']['hits'][0]['fields']['url'][0]

    return body, url

# Run an Elasticsearch query using hybrid RRF scoring of KNN and BM25
@elasticapm.capture_span("knn_search")
def search_knn(query_text, es):
    query = {
        "bool": {
            "must": [{
                "match": {
                    "body_content": {
                        "query": query_text
                    }
                }
            }],
            "filter": [{
              "term": {
                "url_path_dir3": "elasticsearch"
              }
            }]
        }
    }

    knn = [
    {
      "field": "chunk-vector",
      "k": 10,
      "num_candidates": 10,
      "filter": {
        "bool": {
          "filter": [
            {
              "range": {
                "chunklength": {
                  "gte": 0
                }
              }
            },
            {
              "term": {
                "url_path_dir3": "elasticsearch"
              }
            }
          ]
        }
      },
      "query_vector_builder": {
        "text_embedding": {
          "model_id": "sentence-transformers__msmarco-minilm-l-12-v3",
          "model_text": query_text
        }
      }
    }]

    rank = {
       "rrf": {
       }
    }

    fields= [
        "title",
        "url",
        "position",
        "url_path_dir3",
        "body_content"
      ]

    resp = es.search(index=index,
                     query=query,
                     knn=knn,
                     rank=rank,
                     fields=fields,
                     size=10,
                     source=False)

    body = resp['hits']['hits'][0]['fields']['body_content'][0]
    url = resp['hits']['hits'][0]['fields']['url'][0]

    return body, url

def truncate_text(text, max_tokens):
    tokens = text.split()
    if len(tokens) <= max_tokens:
        return text

    return ' '.join(tokens[:max_tokens])

# Generate a response from ChatGPT based on the given prompt
def chat_gpt(prompt, max_tokens=1024, max_context_tokens=4000, safety_margin=5, sys_content=None):

    # Truncate the prompt content to fit within the model's context length
    truncated_prompt = truncate_text(prompt, max_context_tokens - max_tokens - safety_margin)

    # Make the right OpenAI call depending on the API we're using
    if(os.environ["ELASTIC_PROXY"] == "True"):
      response = openai.ChatCompletion.create(model=openai.default_model,
                                              temperature=0,
                                              messages=[{"role": "system", "content": sys_content},
                                                        {"role": "user", "content": truncated_prompt}]
                                              )
    else:
      response = openai.ChatCompletion.create(engine=openai.default_model,
                                              temperature=0,
                                              messages=[{"role": "system", "content": sys_content},
                                                        {"role": "user", "content": truncated_prompt}]
                                              )


    # APM: add metadata labels of data we want to capture
    elasticapm.label(model = openai.default_model)
    elasticapm.label(prompt = prompt)
    elasticapm.label(total_tokens = response["usage"]["total_tokens"])
    elasticapm.label(prompt_tokens = response["usage"]["prompt_tokens"])
    elasticapm.label(response_tokens = response["usage"]["completion_tokens"])
    if 'USER_HASH' in os.environ: elasticapm.label(user = os.environ['USER_HASH'])

    return response["choices"][0]["message"]["content"]

def toLLM(resp, url, usr_prompt, sys_prompt, neg_resp, show_prompt):
    prompt_template = Template(usr_prompt)
    prompt_formatted = prompt_template.substitute(query=query, resp=resp, negResponse=negResponse)
    answer = chat_gpt(prompt_formatted, sys_content=sys_prompt)

    # Display response from LLM
    st.header('Response from LLM')
    st.markdown(answer.strip())

    # We don't need to return a reference URL if it wasn't useful
    if not negResponse in answer:
        st.write(url)

    # Display full prompt if checkbox was selected
    if show_prompt:
        st.divider()
        st.subheader('Full prompt sent to LLM')
        prompt_formatted

# Prompt Defaults
prompt_default = """Answer this question: $query
Using only the information from this Elastic Doc: $resp
Format the answer in complete markdown code format
If the answer is not contained in the supplied doc reply '$negResponse' and nothing else"""

system_default = 'You are a helpful assistant.'
neg_default = "I'm unable to answer the question based on the information I have from Elastic Docs."


''' Main chat form
'''
st.title("ElasticDocs GPT")

with st.form("chat_form"):

    query = st.text_input("Ask the Elastic Documentation a question: ", placeholder='I want to secure my elastic cluster')

    with st.expander("Show Prompt Override Inputs"):
        # Inputs for system and User prompt override
        sys_prompt = st.text_area("create an alernative system prompt", placeholder=system_default, value=system_default)
        usr_prompt = st.text_area("create an alternative user prompt required -> \$query, \$resp, \$negResponse",
                                   placeholder=prompt_default, value=prompt_default )

        # Default Response when criteria are not met
        negResponse = st.text_area("Create an alternative negative response", placeholder = neg_default, value=neg_default)

    show_full_prompt = st.checkbox('Show Full Prompt Sent to LLM')

    # Query Submit Buttons
    col1, col2, col3 = st.columns(3)
    with col1:
        bm25_button = st.form_submit_button("Use BM25")
    with col2:
        knn_button = st.form_submit_button("Use kNN")
    with col3:
        elser_button = st.form_submit_button("Use ELSER")

if elser_button:
    apmclient.begin_transaction("query")
    elasticapm.label(search_method = "elser")
    elasticapm.label(query = query)

    resp, url = search_elser(query, es) # run ELSER query
    toLLM(resp, url, usr_prompt, sys_prompt, negResponse, show_full_prompt)

    apmclient.end_transaction("query", "success")
if knn_button:
    apmclient.begin_transaction("query")
    elasticapm.label(search_method = "knn")
    elasticapm.label(query = query)

    resp, url = search_knn(query, es) # run kNN hybrid query
    toLLM(resp, url, usr_prompt, sys_prompt, negResponse, show_full_prompt)

    apmclient.end_transaction("query", "success")
if bm25_button:
    apmclient.begin_transaction("query")
    elasticapm.label(search_method = "bm25")
    elasticapm.label(query = query)

    resp, url = search_bm25(query, es) # run kNN hybrid query
    toLLM(resp, url, usr_prompt, sys_prompt, negResponse, show_full_prompt)

    apmclient.end_transaction("query", "success")