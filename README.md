# Minimum code to run an LLM model from HuggingFace with OpenVINO

## Programs / Files
|#|file name|description|
|---|---|---|
|1|[download_model.py](download_model.py)|Download a LLM model, and convert it into OpenVINO IR model|
|2|[inference.py](inference.py)|Run an LLM model with OpenVINO. One of the most simple LLM inferencing code with OpenVINO and the `optimum-intel` library.|
|3|[inference-stream.py](inference-stream.py)|Run an LLM model with OpenVINO and `optimum-intel`.<br>Display the answer in streaming mode (word by word).|
|4|[inference-stream-openvino-only.py](inference-stream-openvino-only.py)|Run an LLM model with only OpenVINO.<br>This program doesn't require any DL frameworks such as TF or PyTorch. Also, this program doesn't even use the '`optimum-intel`' library or HuggingFace tokenizers to run. This program uses a simple and dumb tokenizer (that I wrote) instead of HF tokenizers.<br>Try swapping the tokenizer to HF tokenizer in case you see only garbage text from the program (uncomment `AutoTokenizer` and comment out `SimpleTokenizer`)| 
|5|[inference-stream-openvino-only-greedy.py](inference-stream-openvino-only-greedy.py)|Same as program #4 but uses 'greedy decoding' instead of sampling.<br>This program generates fixed output text because it always picks the most probability token ID from the predictions (=greedy decoding).|
|6|[inference-stream-openvino-only-stateless.py](inference-stream-openvino-only-stateless.py)|Same as program #4 but supports **STATELESS** models (which does not use the internal state variables to keep KV-cache values inside of the model) instead of stateful models.|

## How to run

1. Preparation

Note: Converting LLM model requires a large amount of memory (>=32GB).
```sh
python -m venv venv
venv\Scripts\activate
python -m pip install -U pip
pip install -U setuptools wheel
pip install -r requirements.txt
```

2. Download an LLM model and generate OpenVINO IR models
```sh
python download_model.py
```
**Hint**: You can use `optimum-cli` tool to download the models from Huggingface hub, too. You need to install `optimum-intel` Python package to export the model for OpenVINO.  
```sh
optimum-cli export openvino -m TinyLlama/TinyLlama-1.1B-Chat-v1.0 --weight-format int4_asym_g64 TinyLlama-1.1B-Chat-v1.0/INT4
optimum-cli export openvino -m intel/neural-chat-7b-v3 --weight-format int4_asym_g64 neural-chat-7b-v3/INT4
```

3. Run inference
```sh
python inference.py
# or
python inference-stream.py
```

![stream.gif GitHub repository](./resources/stream.gif)


## Official '`optimum-intel`' documents  
Following web sites are also infomative and helpful for `optimum-intel` users.  
- ['optimum-intel' GitGHub Repository](https://github.com/huggingface/optimum-intel)  
- [Detailed description of inference API](https://huggingface.co/docs/optimum/intel/inference)

## Test environment
- Windows 11
- OpenVINO 2023.3.0 LTS
