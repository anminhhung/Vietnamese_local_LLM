import requests
import ray

@ray.remote
def send_query(text):
    resp = requests.get("http://localhost:8000/?query={}".format(text))
    return resp.text


english_texts = [
    "How is convnext different from other CNN networks?",
    "Explain the Convolutional Neural Networks?",
    "Explain Linear regression in the most mathematical way possible?",
    "What is the difference between a fully connected layer and a convolutional layer?",
    "What is the difference between a convolutional layer and a pooling layer in a convolutional neural network?",
    "What is the difference between a convolutional neural network and a recurrent neural network?",
    "How to write a batch handler in ray api?",
    "Who is the most famous person in the world?",
    "What is the best way to learn machine learning?",
]

results = ray.get([send_query.remote(text) for text in english_texts])
print("Result returned:", results)