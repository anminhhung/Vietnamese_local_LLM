import requests

english_text = "MLE là gì?"

print(requests.get(
    "http://localhost:8000/", params={"query": english_text}
).text)