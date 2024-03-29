# Run an LLM chat model only with OpenVINO (supports only the stateful, KV-caching enabled LLM models)
#  - Without 'optimum-intel', 'PyTorch' and HF-Tokenizers.
#  This program uses sampling method to generate the output text.

import numpy as np
import openvino as ov

from simple_tokenizer import SimpleTokenizer
from misc import sampling

model_id = 'TinyLlama/TinyLlama-1.1B-Chat-v1.0'
#model_id = 'Intel/neural-chat-7b-v3'
model_vendor, model_name = model_id.split('/')

model_precision = ['FP16', 'INT8', 'INT4'][2]

print(f'LLM model: {model_id}, {model_precision}')

#from transformers import AutoTokenizer
#tokenizer = AutoTokenizer.from_pretrained(model_id)     # Fast and reliable :-)
tokenizer = SimpleTokenizer(model_vendor, model_name)   # (somewhat) compatible tokenizer with HuggingFace tokenizers (simple, slow, and dumb)

device = 'CPU'
ov_core = ov.Core()
ov_config={"PERFORMANCE_HINT": "LATENCY", "NUM_STREAMS": "1", "CACHE_DIR": "./cache"}
ov_model = ov_core.read_model(model=f'{model_name}/{model_precision}/openvino_model.xml')
print(f'Compiling the model to {device}')
compiled_model = ov.compile_model(ov_model, device, ov_config)
infer_request = compiled_model.create_infer_request()


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
input_ids      = input_tokens.input_ids
attention_mask = input_tokens.attention_mask
position       = input_ids.shape[-1]
position_ids   = np.array([range(position)], dtype=np.int64)
beam_idx       = np.array([0], dtype=np.int32)


# Sampling parameters for generated word
temperature = 1.0
top_p = 0.85
top_k = 10

print(f'Sampling parameters - Temperature: {temperature}, Top-p: {top_p}, Top-k: {top_k}')

prev_output = ''
generated_text_ids = np.array([], dtype=np.int32)
eos_token_id = tokenizer.eos_token_id
num_max_token_for_generation = 300

infer_request.reset_state()                                     # Initialize model internal state

for i in range(num_max_token_for_generation):

    # Run inference (to generate the logits for the next word prediction)
    response = infer_request.infer(inputs={'input_ids':input_ids, 'attention_mask':attention_mask, 'position_ids':position_ids, 'beam_idx':beam_idx})

    logits = response['logits'][0, -1, :]
    sampled_id = sampling(logits, temperature=temperature, top_k=top_k, top_p=top_p, do_sample=True)

    if sampled_id == eos_token_id:
        print('\n*** EOS token detected.')
        break
    generated_text_ids = np.append(generated_text_ids, sampled_id)  # Append the predicted word to the bottom of the generated text ID array
    output_text = tokenizer.decode(generated_text_ids)              # Decode and generate the text from the array of token IDs
    print(output_text[len(prev_output):], end='', flush=True)       # Print only the last generated word
    prev_output = output_text

    # Supply only the last predicted (sampled) word ID as the model input from the 2nd iteration, and the latter
    input_ids      = np.array([[sampled_id]], dtype=np.int64)
    attention_mask = np.array([[1]], dtype=np.int64)
    position_ids   = np.array([[position]], dtype=np.int64)
    beam_idx       = np.array([0], dtype=np.int32)
    position      += 1

print(f'\n\n*** Completed.')
