# Local LLM 

## Setup 

```
pip3 install -r requirements.txt
pip3 install -U "ray[default]"
```

Install Ollama
```
curl -fsSL https://ollama.com/install.sh | sh

```

## Extract feature & Store db 
```
python3 src/llama_index/ingest.py
```

## Using different models
### OpenAI
In ./configs/config_files/model.yaml file
```
SERVICE: openai

...

MODEL_ID: "gpt-3.5-turbo" # or "gpt-4-turbo-preview" pr any other model by openai 


```

run this in script in terminal

```
export OPENAI_API_KEY=XXXXX
```

### Ollama
In ./configs/config_files/model.yaml file
```
SERVICE: ollama
...


MODEL_ID: "ontocord/vistral" # or any other model supported by ollama

```

## Run app 

```
# backend 
serve run localbot_app:app_bot
# UI
streamlit run streamlit_chatbot.py
```

### Ollama support

#### For model supported by ollama
[ollama library](https://ollama.ai/library)

```
ollama pull mistral
```

then change the model_id and basename

```
MODEL_ID: "mistral"
MODEL_BASENAME: ""
```

#### Import from GGUF
Ollama supports importing GGUF models in the Modelfile:

1. Create a file named Modelfile, with a FROM instruction with the local filepath to the model you want to import.
```
FROM ./vicuna-33b.Q4_0.gguf
```

2. Create the model in Ollama
```
ollama create example -f Modelfile
```

3. Run the model
```
ollama run example

```

#### Import from PyTorch or Safetensors
See the guide on importing models for more information.

Customize a prompt
Models from the Ollama library can be customized with a prompt. For example, to customize the llama2 model:

ollama pull llama2
Create a Modelfile:

```
FROM llama2

# set the temperature to 1 [higher is more creative, lower is more coherent]
PARAMETER temperature 1

# set the system message
SYSTEM """
You are Mario from Super Mario Bros. Answer as Mario, the assistant, only.
"""
Next, create and run the model:

ollama create mario -f ./Modelfile
ollama run mario
>>> hi
Hello! It's your friend Mario.

```
