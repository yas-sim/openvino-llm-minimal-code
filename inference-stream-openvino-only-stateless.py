# Run an LLM chat model only with OpenVINO (supports only the STATELESS LLM models)
#  - Without 'optimum-intel', 'PyTorch' and HF-Tokenizers.
#  This program uses sampling method to generate the output text.

import numpy as np
import openvino as ov

from simple_tokenizer import SimpleTokenizer
from misc import sampling

model_id = 'TinyLlama/TinyLlama-1.1B-Chat-v1.0'
#model_id = 'Intel/neural-chat-7b-v3'
model_vendor, model_name = model_id.split('/')

model_precision = ['FP16', 'INT8', 'INT4', 'INT4_stateless'][3]

print(f'LLM model: {model_id}, {model_precision}')

#from transformers import AutoTokenizer
#tokenizer = AutoTokenizer.from_pretrained(model_id)     # Fast and reliable :-)
tokenizer = SimpleTokenizer(model_vendor, model_name)   # (somewhat) compatible tokenizer with HuggingFace tokenizers (simple, slow, and dumb)

import os
import json
with open(os.path.join(model_name, model_precision, 'config.json')) as f:
    config_file = json.load(f)
num_attention_heads = config_file['num_attention_heads']
num_hidden_layers   = config_file['num_hidden_layers']
num_key_value_heads = config_file['num_key_value_heads']

device = 'CPU'
ov_core = ov.Core()
ov_config={"PERFORMANCE_HINT": "LATENCY", "NUM_STREAMS": "1", "CACHE_DIR": "./cache"}
ov_model = ov_core.read_model(model=f'{model_name}/{model_precision}/openvino_model.xml')
print(f'Compiling the model to {device}')
compiled_model = ov.compile_model(ov_model, device, ov_config)
infer_request = compiled_model.create_infer_request()

input_names = [ layer.any_name for layer in ov_model.inputs ]               # Obtain names of the model inputs
key_value_input_names = [key for key in input_names if 'key_values' in key]

# Generate dummy KV-cache tensors for the 1st iteration
inputs = {}
for input_name in key_value_input_names:
    model_input = ov_model.input(input_name)
    shape = model_input.get_partial_shape()
    shape[0] = 1        # batch size
    if shape[2].is_dynamic:
        shape[2] = 0
    else:
        shape[1] = 0
    inputs[input_name] = ov.Tensor(model_input.get_element_type(), shape.get_shape())


question = 'Explain the plot of Cinderella in a sentence.'
question = 'What is the best food in Tokyo?'
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

input_text = prompt_text_tinyllama.format(question=question)
#input_text = prompt_text_neuralchat.format(question=question)
print(f'Question: {question}')

# Tokenize the input text (text -> token IDs)
# - The model input for the 1st iteration
input_tokens = tokenizer(text=input_text, return_tensors='np')
inputs['input_ids']      = input_tokens.input_ids
inputs['attention_mask'] = input_tokens.attention_mask
position                 = inputs['input_ids'].shape[-1]
inputs['position_ids']   = np.arange(position, dtype=np.int64).reshape(1, position)


# Sampling parameters for generated word
temperature = 1.0
top_p = 0.85
top_k = 10

print(f'Sampling parameters - Temperature: {temperature}, Top-p: {top_p}, Top-k: {top_k}')

prev_output = ''
generated_text_ids = np.array([], dtype=np.int32)
eos_token_id = tokenizer.eos_token_id
num_max_token_for_generation = 300

for i in range(num_max_token_for_generation):

    # Run inference (to generate the logits for the next word prediction)
    response = infer_request.infer(inputs)

    logits = response['logits'][0, -1, :]
    sampled_id = sampling(logits, temperature=temperature, top_k=top_k, top_p=top_p, do_sample=True)

    if sampled_id == eos_token_id:
        print('\n*** EOS token detected.')
        break
    generated_text_ids = np.append(generated_text_ids, sampled_id)  # Append the predicted word to the bottom of the generated text ID array
    output_text = tokenizer.decode(generated_text_ids)              # Decode and generate the text from the array of token IDs
    print(output_text[len(prev_output):], end='', flush=True)       # Print only the last generated word
    prev_output = output_text

    # Update KV-cache values with the inference result
    for n in range(num_hidden_layers):
        inputs[f'past_key_values.{n}.key']   = response[f'present.{n}.key']
        inputs[f'past_key_values.{n}.value'] = response[f'present.{n}.value']
    # Supply only the last predicted (sampled) word ID as the model input from the 2nd iteration, and the latter
    inputs['input_ids']      = np.array([[sampled_id]], dtype=np.int64)
    inputs['attention_mask'] = np.array([[1]], dtype=np.int64)
    inputs['position_ids']   = np.array([[position]], dtype=np.int64)
    position += 1

print(f'\n\n*** Completed.')
