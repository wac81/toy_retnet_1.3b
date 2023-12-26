import torch
from retnet.modeling_retnet import RetNetModelWithLMHead
from retnet.configuration_retnet import load_config_from_yaml
from transformers import AutoTokenizer,AutoModelForCausalLM
import glob 
config = load_config_from_yaml('configs/retnet-1.3b.yml')
model = RetNetModelWithLMHead(config)

# model = AutoModelForCausalLM.from_pretrained("/nvme/RetNet/checkpoints/checkpoint-70500")
cp_lst = glob.glob("/nvme/RetNet/checkpoints/checkpoint-*")
cp_lst.sort()
print(cp_lst[-1] + "/pytorch_model.bin")
device=torch.device("cuda:0")

model.load_state_dict(torch.load(cp_lst[-1] + "/pytorch_model.bin",map_location=device))
model.to(device)
model.eval()

tokenizer = AutoTokenizer.from_pretrained("baichuan-inc/baichuan-7B", trust_remote_code=True, use_fast=True)
tokenizer.pad_token_id = tokenizer.unk_token_id = 0 # unk. we want this to be different from the eos token
tokenizer.model_max_length = 2048
# tokenizer.pad_token = tokenizer.eos_token
inp = "公元6世纪新柏拉图主义哲学家，阿摩尼奥斯·赫尔米埃和达马希乌斯的学生,"

# with torch.autocast("cuda"):
while True:
    inp = input("user:")
    inputs = "user:{}\nsystem:".format(inp)
    inputs = tokenizer(inputs, return_tensors='pt')
    inputs = inputs.to('cuda:0')
    # parallel forward
    generated = model.generate(**inputs, parallel_compute_prompt=True, max_new_tokens=128)
    # print(tokenizer.decode(generated.cpu()[0], skip_special_tokens=True))
    res = tokenizer.decode(generated.cpu()[0], skip_special_tokens=True)
    #res = tokenizer.batch_decode(generated.cpu())

    print(''.join(res))
