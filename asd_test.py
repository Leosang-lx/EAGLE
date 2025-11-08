from fastchat.model import get_conversation_template
from fastchat.llm_judge.common import load_questions
import os
from tqdm import tqdm
import torch

from AsyncSD import AsyncSDWrapper
from profiler import prof
from ASDConfig import config as rconfig

server_ip = os.environ['SERVER_IP']

def main():
    assert torch.cuda.is_available(), "CUDA is not available"
    device = torch.device("cuda")
    torch.set_grad_enable(False)

    torch.cuda.set_device(device)

    asd_model = AsyncSDWrapper(
        use_eagle3=True,
        base_model_path=rconfig.base_model_path,
        ea_model_path=rconfig.ea_model_path,
        dtype=torch.float16,
        device=device,
        init_comm=True,
        server_ip=server_ip,
    )
    is_server = asd_model.is_server

    asd_model.eval()

    if not is_server:  # drafter client
        your_message = rconfig.your_message
        if 'vicuna' in rconfig.model_name.lower():
            conv = get_conversation_template("vicuna")
        elif 'llama2' in rconfig.model_name.lower():
            conv = get_conversation_template("llama-2-chat")
            sys_p = "You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe.  Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature.\n\nIf a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information."
            conv.system_message = sys_p
        elif 'llama3' in rconfig.model_name.lower():
            messages = [
                {"role": "system",
                "content": "You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe.  Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature.\n\nIf a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information."},
            ]
        
        if 'llama3' in rconfig.model_name.lower():
            messages.append({
                "role": "user",
                "content": your_message
            })
            prompt = asd_model.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
            input_ids = asd_model.tokenizer([prompt], add_special_tokens=False,).input_ids
        else:
            conv.append_message(conv.roles[0], your_message)
            conv.append_message(conv.roles[1], None)
            prompt = conv.get_prompt() + " "

            input_ids = asd_model.tokenizer([prompt]).input_ids

        print('User Prompt:', prompt)

        # [warm-up]
        if rconfig.warmup:
            cnt = tqdm(range(rconfig.warmup_num), desc='Warm-up') if not is_server else range(rconfig.warmup_num)
            for _ in cnt:
                outputs = run(
                    asd_wrapper=asd_model,

                )

def run(asd_wrapper, input_ids=None, log=False, profiler=None):
    outputs = asd_wrapper.co_generate(
        input_ids=input_ids,
        temperature=rconfig.temperature,
        max_new_token=rconfig.max_new_tokens,
        max_length=rconfig.max_length,
        log=log,
        prof=profiler,
        is_llama3=rconfig.model_name.lower().startswith('llama3'),
    )
    if not asd_wrapper.is_server:
        return outputs

if __name__ == "__main__":
    assert rconfig.mode == 'demo'
    main()

