---
layout: post
title:  "AI Foundation Models"
categories: blog post
---

오늘 AI Foundation Models 수업의 중간고사를 쳤는데, 생각보다 수업에서 배운 모델이 많아서 내용을 간략하게나마 정리해두려고 한다.
(14개 수업에 대한 중간고사였는데 PPT 1400장이 나와서... 아마 완벽하게 정리하는 데에는 이 포스트 하나로 충분하지 않을 것 같다)

Disclaimer: 학부 수업에서 배운 내용을 개인 공부 목적으로 정리한 것이니 오류가 있을 수 있습니다!

 # Positional Encoding
 - Absolute position embedding: sinusoidal (sine), cosine
   > 단점: embedding matrix의 fixed size
   > token 사이의 거리를 정확히 반영하는 것은 아님
 - Relative position embedding: absolute의 대안으로 등장
   > T5 bias: attention의 raw logits ($q \cdot k$) 에 scalar를 곱함. 각 attention head마다 다른 position bias 사용
   > 
   > Rotary position embedding (RoPE): RoFormer에서 사용되는 것으로, token embedding을 복소수(complex numbers)로 나타냄.
   > 
   > Attention with Linear Bias (ALiBi): attention logits에 constant value를 더함. attention 마다 상수 (scalar) m을 곱해줌.

# Transfer Learning
- ELMo
> 등장 이유: ELMo 전으로 등장한 Word2Vec, GloVe, FastText와 같은 word embedding method들은 static embedding이라서 문장에서 단어의 문맥 (context) 같은 걸 표현하지 못함.
> 
> 예컨대, "play"라는 단어는 연극으로 쓰일 수도 있고, 아이들의 놀이라는 뜻으로도 쓰일 수 있는데 이 경우는 그냥 같은 단어로 취급됨.
> 하지만 단어의 올바른 뜻을 반영하기 위해서는 컨텍스트 고려가 필수적임.

> ELMo의 경우 context-dependent embedding이라서 word embedding이 sequence 전체에 대한 함수임.
> ELMo는 가장 대표적으로 Multi-layer Bidirectional LSTM을 사용해서 한 단어 기준으로, 왼쪽 + 오른쪽을 다 살펴봄.

- ULMFit: Universal language model finetuning
> ELMo의 문제점: needs to train task-specific model weights from scratch. task마다 모든 weight를 다 finetune해야한다는 단점이 있음.
>
> 그래서 ULMFit은 Language Model Pretraining > Target Task LM Finetuning > Target Task Classifier Finetuning의 단계를 거침.
> ELMo와 마찬가지로 bidirectional LSTM 구조를 사용함.
>
- GPT-1: Generative Pre-Training
> 대망의 GPT-1. ULMFit와 마찬가지로, LM Finetuning을 하지 않으려고 한 모델이고,
> 좀 특징적인 게 있다면 ELMo, ULMFit은 LSTM을 사용한 것에 비해 GPT-1은 transformer (decoder only)를 사용한 모델이다.
> autoregressive (left to right) language modeling
>
- BERT: Pretraining of Deep Bidirectional Transformers for Language Understanding
> GPT가 decoder only 모델이었던 것에 비해 BERT는 encoder only 모델이다.
> 특징: masking을 사용하는데, 이건 input이 다른 토큰들을 보고 "cheating"할 수 없게 만드는 구조.
> pretraining objective: predict masked token based on the context.
> GPT처럼 BERT 역시 다양한 태스크를 위해 디자인 된 모델로, single sentence classification, sentence pair classification, question answering과 같은 다양한 과제를 수행할 수 있음.
>
> pretraining objectives: 1) masked language modeling 2) next sentence prediction
> 1) masked language modeling: input token의 약 15%를 매스킹하는데 적게 매스킹할 경우 computationally expensive하고 많이 매스킹할 경우 모델이 masked tokens를 제대로 파악하지 못할 수도 있음.
>
> 2) next sentence prediction: 문장 2개가 있는 텍스트를 가지고 두 개로 나눔. 50%의 경우는 두번째 문장을 랜덤 문장으로 바꿈. 
