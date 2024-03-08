import torch
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.llms.huggingface_pipeline import HuggingFacePipeline
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline, BitsAndBytesConfig
from langchain.prompts import PromptTemplate
from langchain.chains.question_answering import load_qa_chain

model_kwargs = {'device': 'cuda'}
embeddings = HuggingFaceEmbeddings(model_kwargs=model_kwargs)

quantization_config = BitsAndBytesConfig(
    load_in_8bit_fp32_cpu_offload=True,
    bnb_4bit_compute_dtype=torch.bfloat16
)

model_name = "mistralai/Mistral-7B-Instruct-v0.2"

qna_prompt_template="""### [INST] Instruction: You will be provided with questions and related data. 
Your task is to find the answers to the questions using the given data. Try your best to give the correct and well formattted answer.
If questions is completley unrelated to the document then return 'Not Enough Information. Please use Google Search Buddy'

{context}

### Question: {question} [/INST]"""


def load_llm(model_name=model_name, quantization_config=quantization_config):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name, device_map='cuda', quantization_config=quantization_config)

    pipe = pipeline("text-generation", model=model, tokenizer=tokenizer, max_new_tokens=512)
    llm = HuggingFacePipeline(pipeline=pipe)
    return llm

def create_qa_chain(llm, prompt_template=qna_prompt_template):
    PROMPT = PromptTemplate(
        template=prompt_template, input_variables=["context", "question"]
    )
    qa_chain = load_qa_chain(llm, chain_type="stuff", prompt=PROMPT)
    return qa_chain
