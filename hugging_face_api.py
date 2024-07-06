## Use this in google colab.

!pip install langchain-huggingface
!pip install huggingface_hub
!pip install transformers
!pip install accelerate
!pip install bitsandbytes
!pip install langchain

from google.colab import userdata
sec_key = userdata.get("hugging_face")
print(sec)


from langchain_huggingface import HuggingFaceEndpoint

from google.colab import userdata
sec_key = userdata.get("HUGGINGFACEHUB_API_TOKEN")
print(sec_key)

import os
os.environ["HUGGINGFACEHUB_API_TOKEN"] = sec_key

repo_id = "mistralai/Mistral-7B-Instruct-v0.2"   # is an instruct fine-tuned version of the Mistral-7B-v0.2 
llm = HuggingFaceEndpoint(repo_if=repo_id, max_length=128, temperature=0.7, token=sec_key)

llm

llm.invoke("what is machine learning")   

repo_id="mistralai/Mistral-7B-Instruct-v0.3"
llm = HuggingFaceEndpoint(repo_if=repo_id, max_length=128, temperature=0.7, token=sec_key)

llm.invoke("what is Generative AI")

from langchain import PromptTemplate, LLMChain
from transformers import pipeline


from langchain import PromptTemplate, LLMChain
question="who won the Cricket world cup in the year 2011?"
template = """Question: {question}
Answer: Let's think step by step"""
prompt = PromptTemplate(template=template, input_variables=["question"])
print(prompt)

llm_chain = LLMChain(llm=llm,prompt=prompt)
print(llm_chain.invoke(question))

from langchain_huggingface import HuggingFacePipeline
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

model_id = "gpt2"
model = AutoModelForCausalLM.from_pretrained(model_id)
tokenizer = AutoTokenizer.from_pretrained(model_id)

pipe= pipeline("text-generation", model=model, tokenizer=tokenizer,max_new_tokens=100)
hf = HuggingFacePipeline(pipeline=pipe)

hf

hf.invoke("what is machine learning")

#use HuggingFacePipelines with gpu
gpu_llm = HuggingFacePipeline.from_model_id(
    model_id="gpt2",
    task="text-generation",
    device=0, @replace with device map=auto to use the accelerate library
    pipeline_kwargs={"max_new_tokens":100},
    )

from langchain_core.prompts import PromptTemplate
template=""" Question:{question}
Answer = Lets think step by step. """
prompt = PromptTemplate.from_template(template)

chain = prompt|gpu_llm

question="What is artifical Intelligence?"
chain.invoke({"question":question})