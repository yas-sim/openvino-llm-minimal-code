# Run an LLM chat model only with OpenVINO (supports only the stateful, KV-caching enabled LLM models)
#  - Without 'optimum-intel', 'PyTorch' and HF-Tokenizers.
#  This program uses 'greedy decoding' to generate the output text.

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
tokenizer = SimpleTokenizer(model_vendor, model_name)               # (somewhat) compatible tokenizer with HuggingFace tokenizers

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

prev_output = ''
generated_text_ids = np.array([], dtype=np.int32)
eos_token_id = tokenizer.eos_token_id
num_max_token_for_generation = 300

infer_request.reset_state()                                     # Initialize model internal state

for i in range(num_max_token_for_generation):

    # Run inference (to generate the logits for the next word prediction)
    response = infer_request.infer(inputs={'input_ids':input_ids, 'attention_mask':attention_mask, 'position_ids':position_ids, 'beam_idx':beam_idx})

    sampled_id = np.argmax(response['logits'][0, -1, :])            # Pick a word ID with greedy decoding
    if sampled_id == eos_token_id:
        print('\n*** EOS token detected.')
        break
    generated_text_ids = np.append(generated_text_ids, sampled_id)  # Append the predicted word to the bottom of the generated text ID array
    output_text = tokenizer.decode(generated_text_ids)              # Decode and generate the text from the array of token IDs
    print(output_text[len(prev_output):], end='', flush=True)       # Print only the last generated word
    prev_output = output_text

    # Supply only the last predicted (sampled) word ID as the model input from the 2nd iteration, and the latter
    # ** This is possible only for the 'stateful' model with KV caching enabled. **
    input_ids      = np.array([[sampled_id]], dtype=np.int64)
    attention_mask = np.array([[1]], dtype=np.int64)
    position_ids   = np.array([[position]], dtype=np.int64)
    beam_idx       = np.array([0], dtype=np.int32)
    position      += 1

print(f'\n\n*** Completed.')
