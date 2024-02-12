# Run an LLM chat model only with OpenVINO
#  - Without 'optimum-intel' and 'PyTorch' (but HF-Tokenizer)

import numpy as np

import openvino as ov
from transformers import AutoTokenizer

model_vendor, model_name = [
    [ 'TinyLlama',  'TinyLlama-1.1B-Chat-v1.0'],
    [ 'Intel',      'neural-chat-7b-v3' ],
][0]

model_precision = ['FP16', 'INT8', 'INT4'][2]

print(f'LLM model: {model_vendor}/{model_name}, {model_precision}')


tokenizer = AutoTokenizer.from_pretrained(f'{model_vendor}/{model_name}')


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

# Tokenize (text -> token IDs)
input_tokens = tokenizer(text=input_text, return_tensors='pt',)
input_ids      = input_tokens.input_ids
attention_mask = input_tokens.attention_mask
position_ids   = np.array([range(input_ids.shape[-1])], dtype=np.int64)
beam_idx       = [1]


def softmax(x):
    x = x - np.max(x, axis=0)
    return np.exp(x) / np.sum(np.exp(x), axis=0)


prev_output = ''
generated_text_ids = np.array([], dtype=np.int32)

# Sampling parameters for generated word
temperature = 1.0
top_p = 0.85
top_k = 10

# limit parameters
temperature = 1.0 if temperature <= 0 else temperature
top_p = max(0.0, min(1.0, top_p))
top_k = max(0.0, top_k)

print(f'Sampling parameters - Temperature: {temperature}, Top-p: {top_p}, Top-k: {top_k}')

eos_token_id = tokenizer.eos_token_id
num_max_token_for_generation = 300

beam_idx = [1]

for i in range(num_max_token_for_generation):

    # Run inference (to generate the logits for the next word prediction)
    infer_request.reset_state()
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
    
    sampled_id = sorted_index[sample]                           # Pick a word ID
    if sampled_id == eos_token_id:
        print('\n*** EOS token detected.')
        break
    generated_text_ids = np.append(generated_text_ids, sampled_id)

    output_text = tokenizer.decode(generated_text_ids)          # Decode and generate the text from the array of token IDs
    print(output_text[len(prev_output):], end='', flush=True)   # Print only the last generated word
    prev_output = output_text

    # Add the last generated token ID to the bottom of the input_ids (, and extend attention mask and position_ids accordingly)
    input_ids = np.append(input_ids, [[sampled_id]], axis=1)
    attention_mask = np.append(attention_mask, [[1]], axis=1)
    position_ids = np.append(position_ids, [[len(position_ids[0])]], axis=1)

print(f'\n\n*** Completed.')