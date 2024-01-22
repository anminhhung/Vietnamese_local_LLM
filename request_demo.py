import requests

english_text = "Deep learning khác machine learning như thế nào?"

response = requests.post(f"http://localhost:8000/api/generate?query={english_text}")
response.raise_for_status()
# for chunk in response.iter_content(chunk_size=None, decode_unicode=True):
#     print(chunk, end="")

print(response.text)
print("NUM TOKENS: ", len(response.text))
    # Dogs are the best.