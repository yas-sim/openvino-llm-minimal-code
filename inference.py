from optimum.intel.openvino import OVModelForCausalLM
from transformers import AutoTokenizer, AutoConfig

model_vendor, model_name = 'TinyLlama', 'TinyLlama-1.1B-Chat-v1.0'
#model_vendor, model_name = 'Intel', 'neural-chat-7b-v3'

model_precision = ['FP16', 'INT8', 'INT4'][2]

print(f'LLM model: {model_vendor}/{model_name}, {model_precision}')

tokenizer = AutoTokenizer.from_pretrained(f'{model_vendor}/{model_name}')
ov_model = OVModelForCausalLM.from_pretrained(
    model_id = f'{model_name}/{model_precision}',
    device='CPU',
    ov_config={"PERFORMANCE_HINT": "LATENCY", "NUM_STREAMS": "1", "CACHE_DIR": ""},
    config=AutoConfig.from_pretrained(f'{model_name}/{model_precision}')
)

# The most simple and naive generation method
prompt_text = '2 + 2 ='
input_tokens = tokenizer(prompt_text, return_tensors='pt')
response = ov_model.generate(**input_tokens, max_new_tokens=2)
response_text = tokenizer.batch_decode(response, skip_special_tokens=True)[0]
print(response_text)


# Generation with a prompt message
question = 'Explain the plot of Cinderella in a sentence.'
prompt_text_tinyllama = f"""\
<|system|>
You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe. If a question does not make any sense or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information.</s>
<|user|>
{question}</s>
<|assistant|>
"""

prompt_text_neuralchat = f"""\
### System:
You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe. If a question does not make any sense or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information.
### User:
{question}
### Assistant:
"""

print('Question:', question)
input_tokens = tokenizer(prompt_text_tinyllama, return_tensors='pt', add_special_tokens=False)
response = ov_model.generate(**input_tokens, max_new_tokens=300, temperature=1.0, do_sample=True, top_k=10, top_p=0.85, repetition_penalty=1.2, num_return_sequences=1)
response_text = tokenizer.decode(response[0], skip_special_tokens=True)
print(response_text.split('[/INST]\n')[-1])             # TinyLlama
#print(response_text.split('### Assistant:\n')[-1])      # NeuralChat
