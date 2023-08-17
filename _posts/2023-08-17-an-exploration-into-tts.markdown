---
layout: post
title:  "An Exploration into TTS"
categories: blog post
---

This summer, I had the chance to learn about `Text-to-Speech (TTS)` through an AI R&D internship. 
One of the first things I did was read two arXiv papers on this topic, and they are as follows:
- `Tacotron: Towards End-to-End Speech Synthesis`: https://arxiv.org/abs/1703.10135.
- `Non-Attentive Tacotron: Robust and Controllable Neural TTS Synthesis Including Unsupervised Duration Modeling`: https://arxiv.org/abs/2010.04301.

For simplicity, I refer to the first paper as `Paper 1` and the second paper as `Paper 2` in this post.

## Comparative Understanding of these Papers

While reading these papers, I focused mostly on the difference between these two models, and classified models into those that use attention and those that do not:
|  | Attention-based models | Nonattentive Models (using duration) |
|-------|--------|--------|
| works by | creating alignment based on previous frame | predicts duration for each input token |
| ends when | judges when to end at each step | decides ending point from the start (by using duration) |
| characteristics | dependent on previous frame, attention fail may occur when ending point erroneously calculated | independent frames (needs extra parameters such as range parameters), target duration may be hard to sample |

To prevent `attention fail` (which is when a speech output never ends (may stutter) or ends too prematurely) from happening, Nonattentive Tacotron (NAT) models emerged as the alternative.
The structure of NAT models is as follows: (note that `Paper 2` has a very straightforward diagram that entails the structure of the model that I have not included in this post for copyright reasons)
- Encoder CBHG (Convolution Bank + Highway Network + Bidirectional GRU)
- Range Parameter
- Duration Predictor
- Gaussian Upsampler
- Positional Encoding
- Decoder
- Postnet

## My Task

With the above as background knowledge, my task in this internship was to develop a metric that could evaluate voice similarity between user-inputted personal finetuning voice and model-outputted voice (which is a speaker in the base model).
### Original Method and Ideation
The original way of evaluating voice similarity had been using mel-spectrogram features and calculating the L2 error between a base model speaker and the finetuning speaker.
I noticed that `speech speed was not controllable` with this method. Regardless of which voice I inputted, the base speaker that may be matched to my voice could be either `too slow` or `too fast`.
This brought my attention to the `duration data` I had available in NAT models. My hypothesis was that if I could map a speaker using `both mel-spectrogram feature and duration`, I would be able to map a base speaker that was closer to the finetuning speaker in both voice feature and speech speed.

### Experimentation
#### What I worked with
Python (for writing all mapping scripts), CUDA, GCP (buckets for uploading & downloading data), Docker, PyTorch & TensorFlow (dependencies)

#### Design of Study
- <b>Finetuning voice</b>: I recorded my own voice with 10 English sentences serviced through the company and preprocessed the data (e.g. clipping pauses and normalizing). Finetuning was done with 2 other English finetuning speakers (speakers also in the base model), so a total of 3 speakers were finetuned.
Note that because these 2 other speakers were also in the base model, exception handling had to be done - for this, please see the <i>Exception Handling</i> bulletpoint.

- <b>Base model</b>: To limit the scope of my project (since it had to be done in a few weeks), I decided to focus on the `vowel duration data`. The reason for this was that I learned that duration for vowels took up the majority of the duration for each sentence. Additionally, I assumed that duration for vowels could influence how accents were determined.
Although I had access to Korean, English, and multilingual (Korean + English) models, I decided to use only the English base speaker model, because the processed Korean corpus text file did not show duration for vowels.
In the end, I used a `base model with around 190 English speakers`, and processed `39 vowels` (note that there are different types of vowels in speech processing, such as a1, a2, ux2).

- <b>Data used</b>: For mel-spectrogram features, I used `average mel-spectrogram features across sentences`. For duration data, I used `average duration data (for each vowel) across sentences`.
The data was stored in binary files using the Python struct package. The structures of the binary files were as follows:
  - <b>mel-spectrogram feature</b>: locale num | locales | speaker count in base model | 22 floats for each speaker, with each float representing average mel-spectrogram feature for all sentences for that speaker
  - <b>duration</b>: vowel count | name of each vowel (eg. a1, a2, ..., ux2) | speaker count in base model | 39 floats for each speaker, with each float representing average duration of that vowel for all sentences for that speaker
  
- <b>Metric calculation</b>: I thought of two different ways of combining mel-spectrogram features and duration: `1) weighted sum` and `2) filtering`.
  - <b>weighted sum</b>: I tried 5 different combinations of weights (listed as (feats, dur)): (1, 0), (0.9, 0.1), (0.8, 0.2), (0.5, 0.5), (0, 1). Besides the very first and last two tuples, I tried combinations with more emphasis on features, because I thought that
  when we listen and (subjectively) judge how similar our real voices are to an AI-outputted voice, we pay closer attention to voice features than speech speed.
  The formula for weighted sum was: <i>feat weight * feature L2 loss between base speaker and finetuning speaker + duration weight * duration L2 loss between base speaker and finetuning speaker</i>.
  The mapping script then outputted the base model speaker with the least error value for the italicized formula.
  - <b>filtering</b>: For this method, I tried two different submethods: `1) output the most similar duration base speaker after choosing 20 base speakers most similar in features`, `2) output the most similar feature base speaker after choosing 20 base speakers most similar in duration`.
    The number 20 was chosen because I had around 200+ speakers in the original dataset I chose before switching to the 190 speaker model, so I would be considering roughly 10% of the entire number of base speakers.

#### Results
I had a total of 4 speakers my voice was mapped to. For simplicity, I refer to these speakers as speaker 1, speaker 2, speaker 3, and speaker 4.
Weighted sum tuples are still noted as (feats, dur) weights.
- weighted sum (1,0): speaker 1
- weighted sum (0.9, 0.1): speaker 2
- weighted sum (0.8, 0.2): speaker 3
- weighted sum (0.5, 0.5): speaker 3
- weighted sum (0, 1): speaker 3
- filtering (20 most similar features ➡️ most similar duration): speaker 3
- filtering (20 most similar durations ➡️ most similar feature): speaker 4

### Discussion of Results (Observations), Areas of Improvement, Feedback Received
- <b>Numeric scale</b>: I found that for the weighted sum method, the mapped speaker converges to speaker 3 as duration weights are increased.
This was found to be an issue of scale at the last minute and could not be fixed, but for future improvements, adjusting scale of both feature and duration error seems very necessary.
- <b>Experimentation with numeric values chosen</b>: This study could most likely be replicated with different numeric values and show different results.
For instance, for the filtering method, instead of looking at the first 20 speakers, I could also look at the first 10 or 30 speakers as appropriate.
Moreover, different weights could be chosen for the weighted sum method. It may be particularly useful to do this over a continuous range of weights to see how the values change.
I thought that more experimentation with different numeric values as input may help develop a more thorough understanding of which numbers are needed for developing the metric.
- <b>Use of vowels</b>: While vowels <i>do</i> take up the majority of a duration of a sentence, there are also some consonants that take up duration of a sentence.
Thus expanding the scope of this project to include both vowels and consonants would capture a more accurate representation of duration data.


