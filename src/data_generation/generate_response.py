from vllm import LLM, SamplingParams
import json

prompt_template = """
    Bạn là một trợ lý ảo hữu ích, bạn sẽ sử dụng ngữ cảnh được cung cấp để trả lời các câu hỏi của người dùng.
    Hãy trả lời câu hỏi sau: "{question}" 
"""


prompt = "Attention là gì?"

def main():
    llm = LLM(model="bkai-foundation-models/vietnamese-llama2-7b-120GB",  trust_remote_code=True, max_model_len=1024)
    sampling_params = SamplingParams(temperature=0.7, top_p=0.95, max_tokens=1024)

    # load instruction from file
    # with open("/home/server1-ailab/Desktop/Bach/Vietnamese_local_LLM/data/augment_ques.json", "r") as f:
    #     prompts = json.load(f)
    
    # for prompt in prompts[:10]:
    outputs =  llm.generate(prompts=prompt_template.format(question=prompt.replace("7.", "").replace("\n", "")), sampling_params=sampling_params)
    for output in outputs:
        print(output.prompt)
        print( output.outputs[0].text )
if __name__ == "__main__":
    main()