# Local LLM 

## Setup 
```
pip3 install -r requirements.txt
pip3 install -U "ray[default]"
```

## Extract feature & Store db 
```
python3 src/ingest.py
```

## Run app 

```
streamlit run streamlit_chatbot.py
```
