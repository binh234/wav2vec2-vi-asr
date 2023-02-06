---
language: vi
datasets:
- vivos
- common_voice
metrics:
- wer
tags:
- audio
- automatic-speech-recognition
- speech
- speechbrain
- Transformer
license: cc-by-nc-4.0
model-index:
- name: Wav2vec2 Base Vietnamese 270h
  results:
  - task: 
      name: Speech Recognition
      type: automatic-speech-recognition
    dataset:
      name: Common Voice vi
      type: common_voice
      args: vi
    metrics:
       - name: Test WER
         type: wer
         value: 9.66
  - task: 
      name: Speech Recognition
      type: automatic-speech-recognition
    dataset:
      name: VIVOS
      type: vivos
      args: vi
    metrics:
       - name: Test WER
         type: wer
         value: 4.04
---
# Wav2Vec2-Base-Vietnamese-270h
Fine-tuned Wav2Vec2 model on Vietnamese Speech Recognition task using about 270h labeled data combined from multiple datasets including [Common Voice](https://huggingface.co/datasets/common_voice), [VIVOS](https://huggingface.co/datasets/vivos), [VLSP2020](https://vlsp.org.vn/vlsp2020/eval/asr). The model was fine-tuned using SpeechBrain toolkit with a custom tokenizer. For a better experience, we encourage you to learn more about [SpeechBrain](https://speechbrain.github.io/).  
When using this model, make sure that your speech input is sampled at 16kHz.  
Please refer to [huggingface blog](https://huggingface.co/blog/fine-tune-wav2vec2-english) or [speechbrain](https://github.com/speechbrain/speechbrain/tree/develop/recipes/CommonVoice/ASR/CTC) on how to fine-tune Wav2Vec2 model on a specific language.

### Tokenizers

```python
from tokenizer import Wav2Vec2WordpieceTokenizer

rhyme_tokenizer = Wav2Vec2WordpieceTokenizer.from_pretrained("/content/virhyme")
syllable_tokenizer = Wav2Vec2WordpieceTokenizer.from_pretrained("/content/visyllable")

text = "mức độ gia tăng dân số trong quý bốn"

print(rhyme_tokenizer.tokenize(text))
# ['m', 'ức', '|', 'đ', 'ộ', '|', 'gi', 'a', '|', 't', 'ăng', '|', 'd', 'ân', '|', 's', 'ố', '|', 'tr', 'ong', '|', 'qu', 'ý', '|', 'b', 'ốn']
print(syllable_tokenizer.tokenize(text))
# ['m', 'ức', '|', 'đ', 'ộ', '|', 'gi', 'a', '|', 't', 'ă', 'ng', '|', 'd', 'â', 'n', '|', 's', 'ố', '|', 'tr', 'o', 'ng', '|', 'q', 'uý', '|', 'b', 'ố', 'n']
```

### Pretrained model
| Data | Model link |
|---|---|
|100h| [viwav2vec2-base-100h](https://huggingface.co/dragonSwing/viwav2vec2-base-100h) |
|1500h| [viwav2vec2-base-1.5k](https://huggingface.co/dragonSwing/viwav2vec2-base-1.5k) |
|3000h| [viwav2vec2-base-3k](https://huggingface.co/dragonSwing/viwav2vec2-base-3k) |

### Fine-tuned model
- Pretrained model: [link](https://huggingface.co/dragonSwing/wav2vec2-base-vn-270h)
- Demonstration: [link](https://huggingface.co/spaces/dragonSwing/wav2vec2-vi-asr)

### Benchmark WER result:
| | [VIVOS](https://huggingface.co/datasets/vivos) | [COMMON VOICE VI](https://huggingface.co/datasets/common_voice) |
|---|---|---|
|without LM| 8.41 | 17.82 |
|with 4-grams LM| 4.04 | 9.66 |

The language model was trained using [OSCAR](https://huggingface.co/datasets/oscar-corpus/OSCAR-2109) dataset on about 30GB of crawled text.

### Install SpeechBrain
To use this model, you should install speechbrain from source. This is not required for speechbrain version > 0.5.10

### Usage
The model can be used directly (without a language model) as follows:
```python
from speechbrain.pretrained import EncoderASR

model = EncoderASR.from_hparams(source="dragonSwing/wav2vec2-base-vn-270h", savedir="pretrained_models/asr-wav2vec2-vi")
model.transcribe_file('dragonSwing/wav2vec2-base-vn-270h/example.wav')
```

### Inference on GPU
To perform inference on the GPU, add  `run_opts={"device":"cuda"}`  when calling the `from_hparams` method.

### Evaluation
The model can be evaluated as follows on the Vietnamese test data of Common Voice.
```python
import torch
import torchaudio
from datasets import load_dataset, load_metric, Audio
from transformers import Wav2Vec2FeatureExtractor
from speechbrain.pretrained import EncoderASR
import re
test_dataset = load_dataset("common_voice", "vi", split="test")
test_dataset = test_dataset.cast_column("audio", Audio(sampling_rate=16_000))
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
wer = load_metric("wer")
extractor = Wav2Vec2FeatureExtractor.from_pretrained("dragonSwing/wav2vec2-base-vn-270h")
model = EncoderASR.from_hparams(source="dragonSwing/wav2vec2-base-vn-270h", savedir="pretrained_models/asr-wav2vec2-vi", run_opts={'device': device})
chars_to_ignore_regex = r'[,?.!\-;:"“%\'�]'
# Preprocessing the datasets.
# We need to read the audio files as arrays
def speech_file_to_array_fn(batch):
  audio = batch["audio"]
  batch["target_text"] = re.sub(chars_to_ignore_regex, '', batch["sentence"]).lower()
  batch['speech'] = audio['array']
  return batch
test_dataset = test_dataset.map(speech_file_to_array_fn)

def evaluate(batch):
  # For padding inputs only
  inputs = extractor(
    batch['speech'], 
    sampling_rate=16000, 
    return_tensors="pt", 
    padding=True, 
    do_normalize=False
  ).input_values
  input_lens = torch.ones(inputs.shape[0])
  pred_str, pred_tokens = model.transcribe_batch(inputs, input_lens)
  batch["pred_strings"] = pred_str
  
  return batch
result = test_dataset.map(evaluate, batched=True, batch_size=4)
print("WER: {:2f}".format(100 * wer.compute(predictions=result["pred_strings"], references=result["target_text"])))
```
**Test Result**: 17.817680%

#### Citation
```
@misc{SB2021,
    author = {Ravanelli, Mirco and Parcollet, Titouan and Rouhe, Aku and Plantinga, Peter and Rastorgueva, Elena and Lugosch, Loren and Dawalatabad, Nauman and Ju-Chieh, Chou and Heba, Abdel and Grondin, Francois and Aris, William and Liao, Chien-Feng and Cornell, Samuele and Yeh, Sung-Lin and Na, Hwidong and Gao, Yan and Fu, Szu-Wei and Subakan, Cem and De Mori, Renato and Bengio, Yoshua },
    title = {SpeechBrain},
    year = {2021},
    publisher = {GitHub},
    journal = {GitHub repository},
    howpublished = {\\\\url{https://github.com/speechbrain/speechbrain}},
  }
```

#### About SpeechBrain
SpeechBrain is an open-source and all-in-one speech toolkit. It is designed to be simple, extremely flexible, and user-friendly. Competitive or state-of-the-art performance is obtained in various domains.
Website: [https://speechbrain.github.io](https://speechbrain.github.io/)
GitHub: [https://github.com/speechbrain/speechbrain](https://github.com/speechbrain/speechbrain)