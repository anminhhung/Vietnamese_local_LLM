import requests

english_text = "Machine learning là gì?"

response = requests.post(f"http://localhost:8000/stream-bot?query={english_text}")
response.raise_for_status()
# for chunk in response.iter_content(chunk_size=None, decode_unicode=True):
#     print(chunk, end="")

print(response.text)
    # Dogs are the best.