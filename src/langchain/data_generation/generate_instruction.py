from vllm import LLM, SamplingParams
import json
from tqdm import tqdm

N_QUESTIONS = 2048
BATCH_SIZE = 8

prompts = [
    "Từ các câu hỏi sau đây, hãy tạo rao những câu hỏi tương tự trong lĩnh vực machine learning, deep learning, AI: \n\n"
    "1. Transformer bao gồm những thành phần chính nào?\n"
    "2. Mô hình BERT có những ứng dụng gì?\n"
    "3. Cơ chế attention hoạt động như thế nào?\n"
    "5. Finetuning khác với transfer learning như thế nào?\n"
    "6. Hãy cho tôi code của một hình VGG đơn giản\n",
]


refine_prompt = [
    "Sử lỗi chính tả trong câu sau: \n"
    "{question}\n"
]

def main():
    llm = LLM(model="vilm/vinallama-7b-chat",  trust_remote_code=True)
    sampling_params = SamplingParams(temperature=1., top_p=0.95)

    augment_ques = []

    for i in tqdm(range(N_QUESTIONS // BATCH_SIZE), total=N_QUESTIONS // BATCH_SIZE):
        outputs =  llm.generate(prompts=prompts * BATCH_SIZE, sampling_params=sampling_params)

        for output in outputs:
            augment_ques.append( output.outputs[0].text )

    # filter "" instrunctiom
    augment_ques = [ques.split("8.")[0] for ques in augment_ques if ques != ""]

    # # refine instruction
    # augment_ques = [refine_prompt[0].format(question=ques) for ques in augment_ques]
    # print(augment_ques)
    # refine_ques = []

    # for i in tqdm(range(N_QUESTIONS // BATCH_SIZE), total=N_QUESTIONS // BATCH_SIZE):
    #     outputs =  llm.generate(prompts=augment_ques[i: i+BATCH_SIZE], sampling_params=sampling_params)

    #     for output in outputs:
    #         refine_ques.append({"instruction": output.outputs[0].text})
    # # Write augment ques to file as json
    # print(refine_ques)

    with open("/home/server1-ailab/Desktop/Bach/Vietnamese_local_LLM/data/augment_ques.json", "w") as f:
        json.dump(augment_ques, f, ensure_ascii=False, indent=4)

if __name__ == "__main__":
    main()