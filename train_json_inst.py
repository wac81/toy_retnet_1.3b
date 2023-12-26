from dataclasses import dataclass

from transformers import (Trainer, TrainingArguments, AutoTokenizer, HfArgumentParser,
                          DataCollatorForLanguageModeling)
from datasets import load_dataset

from retnet.modeling_retnet import RetNetModelWithLMHead
from retnet.configuration_retnet import load_config_from_yaml
import torch
from dataset_json import construct_dataset
import glob

data_config = {
    "mode": "instruct",
    "data": {
        "mixed": "/nvme/Open-Llama/data/instruction_data/part-*.jsonl.zst",
        # "wudao": "/nvme/Open-Llama/data/instruction_data/part-wudao*.jsonl.zst",
        # 由于加载了Llama模型的ckpt所以只使用少量英文数据
        # "the_pile": "/nvme/Open-Llama/data/instruction_data/part-pile-2*.jsonl.zst",
    },
    "pad_to_max": False,
    "sequence_sample_mode": "sample",
    "concat_multiple_sequence": True,
    "num_sequences": 8,
    "seq_length": 512,
}
device=torch.device("cuda:0")


@dataclass
class MyArgs:
    model_size: str = '1.3b'
    dataset_name: str = 'json'
    text_col: str = 'content'
    max_length: int = 1024

def main():
    parser = HfArgumentParser((TrainingArguments, MyArgs))
    
    train_args, args = parser.parse_args_into_dataclasses()
    train_args.save_total_limit = 3
    train_args.gradient_accumulation_steps = 64
    train_args.max_steps = 500000
    train_args.eval_accumulation_steps = 1

    if args.dataset_name == 'json': 
        #sample_by="document" 加载所有文本
        train_dataset = load_dataset('json', data_files={'train': "/nvme/Open-Llama/data/WuDaoCorpus2.0_base_200G/part-2021270707.json"},split='train[5%:95%]') #[5%:95%] 
        eval_dataset = load_dataset('json', 
                                    data_files={'train': "/nvme/Open-Llama/data/WuDaoCorpus2.0_base_200G/part-2021270707.json"},
                                    split='train[:5%]+train[-5%:]')  #train[:5%]+train[-5%:]
    else:
        train_dataset = load_dataset(args.dataset_name, split="train")
        eval_dataset = load_dataset(args.dataset_name, split="validation")

    config = load_config_from_yaml(f"configs/retnet-{args.model_size}.yml")
    model = RetNetModelWithLMHead(config)
    #load
    cp_lst = glob.glob("./checkpoints/*")
    print(cp_lst[-1] + "/pytorch_model.bin")
    model.load_state_dict(torch.load(cp_lst[0] + "/pytorch_model.bin",map_location=device))

    # model.load_state_dict(torch.load(cp_lst[-1] + "/pytorch_model.bin",map_location=device))

    # tokenizer = AutoTokenizer.from_pretrained('gpt2') 
    # tokenizer.model_max_length = 16384
    # tokenizer.pad_token = tokenizer.eos_token
    # tokenizer.unk_token = tokenizer.eos_token
    # tokenizer.bos_token = tokenizer.eos_token

    tokenizer = AutoTokenizer.from_pretrained("baichuan-inc/baichuan-7B", trust_remote_code=True, use_fast=True)
    tokenizer.pad_token_id = tokenizer.unk_token_id = tokenizer.eos_token_id = 0  # unk. we want this to be different from the eos token
    # tokenizer.pad_token = tokenizer.eos_token
    # tokenizer.unk_token = tokenizer.eos_token
    # tokenizer.bos_token = tokenizer.eos_token


    def tokenize_datset(example):
        # example[args.text_col] = [line for line in example[args.text_col] if len(line) > 0 and not line.isspace()]
        # if len(example[args.text_col]) == 0: return False

        input_ids = tokenizer(example[args.text_col],
                              truncation=True,
                              max_length=args.max_length,
                            #   padding="max_length",
                              return_tensors='pt').input_ids[0]
        return {'input_ids': input_ids}

    train_dataset = train_dataset.filter(lambda x: len(x["content"]) > 5,num_proc=8)
    eval_dataset = eval_dataset.filter(lambda x: len(x["content"]) > 5,num_proc=8)

    # train_dataset = train_dataset.map(tokenize_datset, remove_columns=train_dataset.column_names, num_proc=8)
    eval_dataset = eval_dataset.map(tokenize_datset, remove_columns=eval_dataset.column_names, num_proc=8)


    # model.load_state_dict(torch.load("/nvme/RetNet/checkpoints/checkpoint-70500/pytorch_model.bin"))


    train_dataset = construct_dataset(data_config, tokenizer)
    # train_dataset = split_dataset_by_node(
    #     train_dataset,
    #     rank=accelerator.process_index,
    #     world_size=accelerator.num_processes,
    # )
    # train_loader = DataLoader(
    #     train_dataset,
    #     batch_size=config["train"]["train_batch_size"],
    #     num_workers=config["train"]["train_num_workers"],
    #     prefetch_factor=config["train"].get("prefetch_factor", 2),
    #     pin_memory=True,
    # )

    trainer = Trainer(model=model,
                      args=train_args,
                      train_dataset=train_dataset,
                      eval_dataset=eval_dataset,
                      tokenizer=tokenizer,
                      data_collator=DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False))

    if train_args.do_train:
        trainer.train()
        trainer.save_model()
    if train_args.do_eval:
        trainer.evaluate()


if __name__ == "__main__":
    main()

