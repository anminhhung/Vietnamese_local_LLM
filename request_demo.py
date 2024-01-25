import requests

english_text = "Autoencoder là gì?"

# response = requests.post(f"http://localhost:8000/api/generate?query={english_text}")
# response.raise_for_status()
# # for chunk in response.iter_content(chunk_size=None, decode_unicode=True):
# #     print(chunk, end="")

# print(response.text)
# print("NUM TOKENS: ", len(response.text))
#     # Dogs are the best.


response = requests.post(f"http://localhost:8000/api/stream?query={english_text}", stream=True)
response.raise_for_status()
for chunk in response.iter_content(chunk_size=None, decode_unicode=True):
    print(chunk, end="")

    # Dogs are the best.