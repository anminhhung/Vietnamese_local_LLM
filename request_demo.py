import requests

english_text = "cán bộ, nhân viên tham gia chương trình đào tạo cần làm gì?"

# response = requests.post(f"http://localhost:8000/api/generate?query={english_text}")
# # response.raise_for_status()
# # for chunk in response.iter_content(chunk_size=None, decode_unicode=True):
# #     print(chunk, end="")

# print(response.text)
# print("NUM TOKENS: ", len(response.text))
#     # Dogs are the best.


response = requests.post(f"http://127.0.0.1:8000/api/generate?query={english_text}", stream=False)
# response.raise_for_status()
# for chunk in response.iter_content():
    # print(chunk, end="")
print(response.text)
