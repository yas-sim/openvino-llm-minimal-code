# Minimum code to run an LLM model from HuggingFace with OpenVINO

## Programs / Files
|#|file name|description|
|---|---|---|
|1|download_model.py|Download a LLM model, and convert it into OpenVINO IR model|
|2|inference.py|Run an LLM model with OpenVINO.|
|3|inference-stream.py|Run an LLM model with OpenVINO. Display the answer in streaming mode (word by word).|

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

3. Run inference
```sh
python inference.py
# or
python inference-stream.py
```

![stream.gif](./resources/stream.gif)

## Test environment
- Windows 11
- OpenVINO 2023.3.0 LTS
