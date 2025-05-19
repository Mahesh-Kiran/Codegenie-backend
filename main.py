from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL_NAME = "deepseek-ai/deepseek-coder-1.3b-instruct"

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
    device_map="auto"
).eval()

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class CodeRequest(BaseModel):
    prompt: str
    max_tokens: int = 1000

@app.post("/generate")
async def generate_code(request: CodeRequest):
    inputs = tokenizer(request.prompt, return_tensors="pt").to(model.device)
    outputs = model.generate(
        **inputs,
        max_length=request.max_tokens,
        temperature=0.2,
        do_sample=True,
        pad_token_id=tokenizer.eos_token_id
    )
    full_output = tokenizer.decode(outputs[0], skip_special_tokens=True)
    response = full_output[len(request.prompt):].strip()
    return {"response": response}

@app.post("/debug")
async def debug_code(request: CodeRequest):
    code_to_debug = request.prompt.strip()
    enhanced_prompt = (
        "Analyze the following code. "
        "List ONLY the syntax or logical errors (if any) found in the code. "
        "If the code is correct, reply with 'No errors found.'\n\n"
        f"Code:\n{code_to_debug}"
    )
    inputs = tokenizer(enhanced_prompt, return_tensors="pt").to(model.device)
    outputs = model.generate(
        **inputs,
        max_length=request.max_tokens,
        temperature=0.3,
        do_sample=True,
        pad_token_id=tokenizer.eos_token_id,
        eos_token_id=tokenizer.eos_token_id
    )
    full_output = tokenizer.decode(outputs[0], skip_special_tokens=True)
    response = full_output[len(enhanced_prompt):].strip() if full_output.startswith(enhanced_prompt) else full_output
    return {"response": response}
