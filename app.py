from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from langchain.embeddings import SelfHostedEmbeddings
import runhouse as rh
import torch
import json

gpu = rh.cluster(name="rh-a10x", instance_type="A100:1", use_spot=False)


# Init is ran on server startup
# Load your model to GPU as a global variable here using the variable name "model"
def init():
    global model
    
    device = 0 if torch.cuda.is_available() else -1
    model = pipeline('fill-mask', model='bert-base-uncased', device=device)


# # Inference is ran for every server call
# # Reference your preloaded global model variable here.
def inference(model_inputs:dict) -> dict:
    global model

    # Parse out your arguments
    prompt = model_inputs.get('prompt', None)
    if prompt == None:
        return {'message': "No prompt provided"}
    
    # Run the model
    # def get_pipeline():
    #     model_id = "bert-base-uncased"
    #     tokenizer = AutoTokenizer.from_pretrained(model_id)
    #     model = AutoModelForCausalLM.from_pretrained(model_id)
    #     return pipeline("Fill-Mask", model=model, tokenizer=tokenizer)
    

    def inference_fn(pipeline, prompt):
    # Return last hidden state of the model
     if isinstance(prompt, list):
        return [emb[0][-1] for emb in pipeline(prompt)] 
     return pipeline(prompt)[0][-1]


    embeddings = SelfHostedEmbeddings(
    model_load_fn=model, 
    hardware=gpu,
    model_reqs=["./", "torch", "transformers"],
    inference_fn=inference_fn)

    result = embeddings.flatten('F')
    embd = json.dumps(result.tolist())
    return {'data': embd}
    
