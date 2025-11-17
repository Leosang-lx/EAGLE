from eagle.model.ea_model import EaModel
from fastchat.model import get_conversation_template
import torch
from transformers import AutoConfig
import time
from ASDConfig import config as rconfig
from tqdm import tqdm
from profiler import prof
from utils import prof_or_null
# from eagle.model.utils import calculate_model_size_with_buffers

device = 0
torch.cuda.set_device(device) 

base_model_path = rconfig.base_model_path
EAGLE_model_path = rconfig.ea_model_path

config = AutoConfig.from_pretrained(base_model_path)

# print(config)
# exit(0)

model = EaModel.from_pretrained(
    base_model_path=base_model_path,
    ea_model_path=EAGLE_model_path,
    torch_dtype=torch.float16,
    # low_cpu_mem_usage=True,
    device_map=f"cuda:{device}",
    total_token=rconfig.total_token,
    depth=rconfig.depth,
)

# show memory usage
show_mem = False
if show_mem:
    # model_size = calculate_model_size_with_buffers(model)
    # print(f"Model size: {model_size:.2f} MB")
    print(torch.cuda.list_gpu_processes(device=f"cuda:1"))
    mem_allocated_line = "Memory allocated:" + str(torch.cuda.memory_allocated(device=f"cuda:1") / (1024 * 1024)) + " MB"
    max_memory_allocated_line = "Max memory usage:" + str(torch.cuda.max_memory_allocated(device=f"cuda:1") / (1024 * 1024)) + " MB"
    mem_reserved_line = "Memory reserved:" + str(torch.cuda.memory_reserved(device=f"cuda:1") / (1024 * 1024)) + " MB"
    max_memory_reserved_line = "Max memory reserved:" + str(torch.cuda.max_memory_reserved(device=f"cuda:1") / (1024 * 1024)) + " MB"
    prof_lines = [mem_allocated_line, max_memory_allocated_line, mem_reserved_line, max_memory_reserved_line]
    print('\n'.join(prof_lines) + '\n')


model.eval()


# [prompt and template]
your_message = rconfig.your_message
# your_message = 'Who are you?'
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
    prompt = model.tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    input_ids = model.tokenizer([prompt], add_special_tokens=False,).input_ids
else:
    conv.append_message(conv.roles[0], your_message)
    conv.append_message(conv.roles[1], None)
    prompt = conv.get_prompt() + " "

    input_ids = model.tokenizer([prompt]).input_ids

input_ids = torch.as_tensor(input_ids).cuda()

print('\n=========PROMPT=========')
print(prompt)

# input_ids=model.tokenizer([prompt]).input_ids
print('Input lenght:', input_ids.size(-1))
# input_ids = torch.as_tensor(input_ids).cuda()


## warmup
for w in tqdm(range(rconfig.warmup)):
    _ = model.eagenerate(input_ids, temperature=0.0, max_new_tokens=512, log=False)

### test
for t in tqdm(range(rconfig.test)):
    if rconfig.use_spec:
        with prof_or_null('eagenerate', prof):
            output_ids, new_token, idx = model.eagenerate(input_ids,temperature=0.0,max_new_tokens=512, log=True, prof=prof)
    else:  # autoregressive
        with prof_or_null('naivegenerate', prof):
            output_ids, new_token, idx = model.naivegenerate(input_ids,temperature=0.0,max_new_tokens=512, log=True)


if show_mem:
    torch.cuda.empty_cache()
    print(torch.cuda.list_gpu_processes(device=f"cuda:1"))
    # mem_summary = torch.cuda.memory_summary() + "\n"
    # mem_state = torch.cuda.memory_stats(device=f"cuda:1")
    mem_allocated_line = "Memory allocated:" + str(torch.cuda.memory_allocated(device=f"cuda:1") / (1024 * 1024)) + " MB"
    max_memory_allocated_line = "Max memory usage:" + str(torch.cuda.max_memory_allocated(device=f"cuda:1") / (1024 * 1024)) + " MB"
    mem_reserved_line = "Memory reserved:" + str(torch.cuda.memory_reserved(device=f"cuda:1") / (1024 * 1024)) + " MB"
    max_memory_reserved_line = "Max memory reserved:" + str(torch.cuda.max_memory_reserved(device=f"cuda:1") / (1024 * 1024)) + " MB"
    prof_lines = [mem_allocated_line, max_memory_allocated_line, mem_reserved_line, max_memory_reserved_line]
    print('\n'.join(prof_lines) + '\n')

print('New tokens:', new_token)
output=model.tokenizer.decode(output_ids[0][-new_token:])
print('Rounds:', idx+1)
# print('Reach leaf:', reach_leaf)

print('\n=========OUTPUT=========')
print(output)

prof.print_all_elapsed_times()