# Run an LLM chat model only with OpenVINO (no KV-caching enabled)
#
# *** VERY SLOW VERSION *** 
# *** This program is developed just to demonstrate how much KV-caching improves the LLM inference performance

import numpy as np
import openvino as ov

from simple_tokenizer import *
from misc import softmax

model_vendor, model_name = [
    [ 'TinyLlama',  'TinyLlama-1.1B-Chat-v1.0'],
    [ 'Intel',      'neural-chat-7b-v3' ],
][0]

model_precision = ['FP16', 'INT8', 'INT4'][2]

print(f'LLM model: {model_vendor}/{model_name}, {model_precision}')

#from transformers import AutoTokenizer
#tokenizer = AutoTokenizer.from_pretrained(f'{model_vendor}/{model_name}')
tokenizer = SimpleTokenizer(model_vendor, model_name)

device = 'CPU'
ov_core = ov.Core()
ov_config={"PERFORMANCE_HINT": "LATENCY", "NUM_STREAMS": "1", "CACHE_DIR": ""}
ov_model = ov_core.read_model(model=f'{model_name}/{model_precision}/openvino_model.xml')
print(f'Compiling the model to {device}')
compiled_model = ov.compile_model(ov_model, device, ov_config)
infer_request = compiled_model.create_infer_request()


input_text = 'Explain the plot of Cinderella in a sentence. \n\n'
input_text = 'What is the best food in Tokyo? \n\n'
print(f'Input text: {input_text}')

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

# Limit the range of sampling parameters
temperature = 1.0 if temperature <= 0 else temperature
top_p = max(0.0, min(1.0, top_p))
top_k = max(0.0, top_k)

print(f'Sampling parameters - Temperature: {temperature}, Top-p: {top_p}, Top-k: {top_k}')

prev_output = ''
generated_text_ids = np.array([], dtype=np.int32)
eos_token_id = tokenizer.eos_token_id
num_max_token_for_generation = 300

infer_request.reset_state()                                     # Initialize model internal state

for i in range(num_max_token_for_generation):

    # Run inference (to generate the logits for the next word prediction)
    infer_request.reset_state()                                 # Need to reset the internal state on every iteration (just wasting the KV cache and reconstruct the KV value on every iterations)
    response = infer_request.infer(inputs={'input_ids':input_ids, 'attention_mask':attention_mask, 'position_ids':position_ids, 'beam_idx':beam_idx})

    # Basic post process (logits->probabilities, sort)
    next_token_prob = softmax(response['logits'][0, -1, :])     # Apply softmax
    next_token_prob /= temperature                              # Scale probabilities by 'temperature' parameter
    sorted_index = np.argsort(next_token_prob)[::-1]            # Sort probability and generate an array of indices

    # Top-p
    sum_prob = 0
    top_p_num = 0
    for top_p_num in range(len(sorted_index)):
        sum_prob += next_token_prob[sorted_index[top_p_num]]    # Accumulate the probability values
        top_p_num += 1
        if sum_prob >= top_p:                                   # Break when the accumlated probability exceeds the top-p value
            break

    # Top-k
    top_k_num = top_k if top_k <= top_p_num else top_p_num      # Limit the samples by top-k

    rand = np.random.rand() * top_p                             # Generate a random value for sampling (range = 0.0 ~ top_p)
    sum_prob = 0
    for sample in range(top_k_num):
        sum_prob += next_token_prob[sorted_index[sample]]       # Accumulate the probability value
        if sum_prob >= rand:                                    # Break when the accumulated probability exceeds sampling target value
            break
    
    sampled_id = sorted_index[sample]                           # Pick a word ID (= predicted next word ID)
    if sampled_id == eos_token_id:
        print('\n*** EOS token detected.')
        break
    generated_text_ids = np.append(generated_text_ids, sampled_id)  # Append the predicted word to the bottom of the generated text ID array
    output_text = tokenizer.decode(generated_text_ids)              # Decode and generate the text from the array of token IDs
    print(output_text[len(prev_output):], end='', flush=True)       # Print only the last generated word
    prev_output = output_text

    # Append the last generated word ID to the bottom of the input_ids 
    input_ids      = np.append(input_ids, [[sampled_id]], axis=1)
    attention_mask = np.append(attention_mask, [[1]], axis=1)
    position_ids   = np.append(position_ids, [[position]], axis=1)
    beam_idx       = np.array([0], dtype=np.int32)
    position      += 1

print(f'\n\n*** Completed.')
