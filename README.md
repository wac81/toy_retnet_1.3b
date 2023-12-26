## 介绍 (Introduction)
retnet-1.3B-toy 是一个开源模型。
1.根据retnet论文([https://arxiv.org/pdf/2307.08621.pdf](https://arxiv.org/pdf/2307.08621.pdf))开发并基于transformer文本生成模型。该仓库的算法实现根据repo进行([https://github.com/syncdoth/RetNet.git](https://github.com/syncdoth/RetNet.git))
2.该仓库目标是建议一个retnet基础训练仓库，建议做学习研究使用，不建议商用。
2.该仓库只使用wiki文本和少量sharegpt/belle/多轮指令数据集训练而成。包含中英文数据，数据估算占比7:3。
3.本次放出pretrain模型与sft微调后模型。
4.已知问题：
  -会出现重复句子回答，可以调节topk减轻该问题。
  -会出现回答不全问题，可以提高max_new_token缓解该问题。
  -由于知识储备不足，回答准确性一般。

retnet-1.3B-toy is an open source model.
1. Developed according to retnet paper ([https://arxiv.org/pdf/2307.08621.pdf](https://arxiv.org/pdf/2307.08621.pdf)) and based on transformer text generation model. The algorithmic implementation of this repository is carried out according to repo ([https://github.com/syncdoth/RetNet.git](https://github.com/syncdoth/RetNet.git))
2. The goal of this repository is to suggest a retnet base training repository, which is recommended to be used for learning research and not for commercial use.
2. This repository is trained using only wiki text and a small amount of sharegpt/belle instruction dataset.
3. This release pretrain model with sft fine-tuned model.
4. known issues:
-Repeated sentence answers will occur, topk can be adjusted to mitigate the problem.
-Incomplete answers will occur, you can increase max_new_token to alleviate the problem.
-Answer accuracy is average due to insufficient knowledge base.

## 软件依赖 (Dependencies)

```shell
pip install torch transformers
```

## 代码调用 (Code Usage)

可以通过如下代码调用 retnet-1.3B-toy 模型来生成对话：

You can generate dialogue by invoking the retnet-1.3B-toy model with the following code:

python generate.py

```shell
user:中医如何医治风寒
system:中医的治疗方法主要包括针灸、针灸、推拿、太极拳等。针灸可以帮助人体解毒、调节身体温度，针灸可以刺激人体的血液循环，推拿可以促进血液循环，推拿可以促进血液循环，从而缓解身体不适。针灸可以帮助人体解毒、调节身体温度，推拿可以促进血液循环，从而缓解身体不适。太极拳则可以帮助人体解毒、调节身体温度，推拿可以促进血液循环，从而缓解身体不适。太极拳则可以帮助人体解毒、调节身体温度，推拿可以促进血液循环，
```


## 协议 (License)

本仓库的代码依照 [Apache-2.0](LICENSE) 协议开源，retnet-1.3B-toy 模型的权重的使用则需要遵循 [Model License](MODEL_LICENSE)。

The code in this repository is open-sourced under the [Apache-2.0 license](LICENSE), while the use of the retnet-1.3B-toy model weights needs to comply with the [Model License](MODEL_LICENSE).
