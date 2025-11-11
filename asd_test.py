### SINGLE REQUEST test: 1*SERVER + 1*CLIENT

from fastchat.model import get_conversation_template
from fastchat.llm_judge.common import load_questions
import os
from tqdm import tqdm
import torch
import multiprocessing as mp

from AsyncSD import AsyncSDWrapper, load_draft_model
from profiler import prof
from ASDConfig import config as rconfig
from utils import prof_or_null

def main(is_server=None):
    assert torch.cuda.is_available(), "CUDA is not available"
    device = 0
    torch.set_grad_enabled(False)

    torch.cuda.set_device(device)

    asd_model = AsyncSDWrapper(
        use_eagle3=True,
        base_model_path=rconfig.base_model_path,
        ea_model_path=rconfig.ea_model_path,
        init_comm=True,
        server_ip=server_ip,
        torch_dtype=torch.float16,
        # use_safetensors=True,
        device_map=f'cuda:{device}',
        is_server=is_server,
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
        
        input_ids = torch.as_tensor(input_ids).cuda()
        print(f'prompt: {prompt}')
        print('input_len:', input_ids.size(-1))

    # [warm-up]
    if rconfig.warmup:
        cnt = tqdm(range(rconfig.warmup), desc='Warm-up') if not is_server else range(rconfig.warmup)
        for _ in cnt:
            outputs = run(
                asd_model,
                input_ids if not is_server else None,
            )
    
    # [test generation]
    profiler = prof if rconfig.prof else None
    cnt = tqdm(range(rconfig.test), desc='Test') if not is_server else range(rconfig.test)
    for i in cnt:
        name = 'SERVER' if is_server else 'CLIENT'
        with prof_or_null(f'{name} co_generate', profiler):
            outputs = run(
                asd_model,
                input_ids if not is_server else None,
                # rconfig.log if not is_server else False,
                rconfig.log,
                profiler,
            )
    # [show]
    if not is_server:
        if rconfig.log:
            output_ids, new_tokens, spec_rounds, turns = outputs
        else:
            output_ids = outputs
        output = asd_model.tokenizer.decode(output_ids[0][-new_tokens:])
        print('Generated:', output)
        if rconfig.log:
            print('New tokens:', new_tokens)
            print('Spec rounds:', spec_rounds)
            print('Turns:', turns)


    prof.print_all_events()

    asd_model.comm.stop()

def run(asd_wrapper, input_ids=None, log=False, profiler=None):
    outputs = asd_wrapper.co_generate(
        input_ids=input_ids,
        temperature=rconfig.temperature,
        max_new_tokens=rconfig.max_new_tokens,
        log=log,
        prof=profiler,
        is_llama3='llama3' in rconfig.model_name.lower(),
    )
    if not asd_wrapper.is_server:
        return outputs

if __name__ == "__main__":
    assert rconfig.mode == 'demo'
    server_ip = '172.18.36.132'

    if server_ip not in ['localhost', '127.0.0.1']:
        # distributed test
        main()

    else:
        # localtest
        drafter_client = mp.Process(target=main, args=(False,))
        verifier_server = mp.Process(target=main, args=(True,))

        drafter_client.start()
        verifier_server.start()

        drafter_client.join()
        verifier_server.join()
