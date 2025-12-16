import os
import torch
import runpod
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

# ==============================================================================
# 1. CONFIGURATION
# ==============================================================================

# This path is where RunPod mounts your Network Volume (ArabicLLMcontainer)
MODEL_PATH = "/runpod-volume/FanarModel"

# Your specific Arabic system prompt
SYSTEM_PROMPT = (
    "أنت مساعد مفيد وذكي. أجب على السؤال المطلوب فقط وبشكل مباشر. "
    "لا تخرج عن سياق السؤال المطروح. كن مختصراً ومفيداً ولا تطل في الحديث دون داعٍ."
)

# Global variables to store the model and tokenizer in memory
tokenizer = None
model = None

# ==============================================================================
# 2. MODEL LOADING (Runs once when worker starts)
# ==============================================================================

def load_model():
    global tokenizer, model
    
    if model is None:
        print(f"--- Loading Fanar model from: {MODEL_PATH} ---")
        
        # Verify the path exists to avoid silent failures
        if not os.path.exists(MODEL_PATH):
            print(f"ERROR: Model path {MODEL_PATH} not found. Check your volume mount!")
            return

        # 4-bit Quantization Configuration
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
        )

        try:
            # Load tokenizer from volume
            tokenizer = AutoTokenizer.from_pretrained(
                MODEL_PATH, 
                local_files_only=True
            )
            
            # Load model from volume into GPU
            model = AutoModelForCausalLM.from_pretrained(
                MODEL_PATH,
                quantization_config=bnb_config,
                device_map="auto",
                local_files_only=True,
                trust_remote_code=True
            )
            
            if not tokenizer.pad_token:
                tokenizer.add_special_tokens({'pad_token': '[PAD]'})
                
            print("--- Model successfully loaded from volume into GPU ---")
            
        except Exception as e:
            print(f"CRITICAL ERROR during model loading: {str(e)}")

# ==============================================================================
# 3. HANDLER FUNCTION (Runs for every request)
# ==============================================================================

def handler(job):
    """
    Standard RunPod handler.
    Input format: {"input": {"prompt": "سؤالك هنا"}}
    """
    job_input = job.get("input", {})
    user_prompt = job_input.get("prompt")
    
    if not user_prompt:
        return {"error": "No prompt provided in the input."}

    # Prepare message structure with System Prompt
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt}
    ]

    # Format the input using the model's chat template
    input_text = tokenizer.apply_chat_template(
        messages, 
        tokenize=False, 
        add_generation_prompt=True
    )
    
    inputs = tokenizer(input_text, return_tensors="pt").to(model.device)

    # Generate response
    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=512,
            do_sample=False,        # Greedy search for more direct answers
            repetition_penalty=1.2,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    # Decode and return only the new text
    full_text = tokenizer.decode(output[0], skip_special_tokens=True)
    # Extract only the assistant's part (everything after the prompt)
    # For a cleaner response, we decode just the generated tokens:
    response = tokenizer.decode(
        output[0][inputs.input_ids.shape[-1]:], 
        skip_special_tokens=True
    ).strip()

    return {"response": response}

# ==============================================================================
# 4. EXECUTION
# ==============================================================================

# Pre-load model so the worker is ready before taking jobs
load_model()

# Start the RunPod serverless worker
runpod.serverless.start({"handler": handler})
