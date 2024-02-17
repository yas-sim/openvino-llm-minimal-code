import numpy as np

def softmax(x):
    x = x - np.max(x, axis=0)
    return np.exp(x) / np.sum(np.exp(x), axis=0)

def sampling(logits, temperature=1.0, top_k=10, top_p=0.85, do_sample=False):

    # Basic post process (logits->probabilities)
    next_token_prob = softmax(logits)                               # Apply softmax
    
    if do_sample:
        # Limit the range of sampling parameters
        temperature = 1.0 if temperature <= 0 else temperature
        top_p = max(0.0, min(1.0, top_p))                           # 0.0 <= top_p <= 1.0
        top_k = max(1.0, top_k)                                     # top_k >= 1.0

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
        top_k_num = int(top_k if top_k <= top_p_num else top_p_num) # Limit the samples by top-k

        rand = np.random.rand() * top_p                             # Generate a random value for sampling (range = 0.0 ~ top_p)
        sum_prob = 0
        for sample in range(top_k_num):
            sum_prob += next_token_prob[sorted_index[sample]]       # Accumulate the probability value
            if sum_prob >= rand:                                    # Break when the accumulated probability exceeds sampling target value
                break
        
        sampled_id = sorted_index[sample]                           # Pick a word ID (= predicted next word ID)
    else:
        sampled_id = np.argmax(next_token_prob)                     # Pick the most high probability word ID (=greedy search)

    return sampled_id
