from optimum.intel.openvino import OVModelForCausalLM
from transformers import AutoTokenizer, AutoConfig

model_id = 'TinyLlama/TinyLlama-1.1B-Chat-v1.0'
#model_id = 'Intel/neural-chat-7b-v3'
model_vendor, model_name = model_id.split('/')

model_precision = ['FP16', 'INT8', 'INT4'][2]

print(f'LLM model: {model_id}, {model_precision}')

tokenizer = AutoTokenizer.from_pretrained(model_id)
ov_model = OVModelForCausalLM.from_pretrained(
    model_id = f'{model_name}/{model_precision}',   # <- OpenVINO model directory. This directory must contain 'openvino_model[.xml|.bin]' and 'config.json'.
    device='CPU',
    ov_config={"PERFORMANCE_HINT": "LATENCY", "NUM_STREAMS": "1", "CACHE_DIR": ""},
    config=AutoConfig.from_pretrained(model_id)
)

# The most simple and naive generation method
prompt_text = 'Explain the plot of Cinderella in a sentence.\n\n'
input_tokens = tokenizer(prompt_text, return_tensors='pt')
response = ov_model.generate(**input_tokens, max_new_tokens=300, num_return_sequences=1, temperature=1.0, do_sample=True, top_k=5, top_p=0.85, repetition_penalty=1.2)
response_text = tokenizer.decode(response[0], skip_special_tokens=True)
print(response_text)
