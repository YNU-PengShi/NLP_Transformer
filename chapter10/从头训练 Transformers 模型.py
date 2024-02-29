import keyword

from transformers import pipeline, set_seed
from datasets import load_dataset, DownloadConfig
import os
import psutil
from transformers import AutoTokenizer
from transformers.models.gpt2.tokenization_gpt2
import byte
from tqdm.auto import tqdm
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
from argparse import Namespace

generation_gpt = pipeline("text-generation", model="openai-gpt")
generation_gpt2 = pipeline("text-generation", model="gpt2")


def model_size(model):
    return sum(t.numel() for t in model.parameters())


print(f"GPT size: {model_size(generation_gpt.model) / 1000 ** 2:.1f}M parameters")
print(f"GPT2 size: {model_size(generation_gpt2.model) / 1000 ** 2:.1f}M parameters")


def enum_pipeline_ouputs(pipe, prompt, num_return_sequences):
    out = pipe(prompt, num_return_sequences=num_return_sequences, clean_up_tokenization_spaces=True)
    return "\n".join(f"{i + 1}." + s["generated_text"] for i, s in enumerate(out))


prompt = "\nWhen they came back"
print("GPT completions:\n" + enum_pipeline_ouputs(generation_gpt, prompt, 3))
print("")
print("GPT-2 completions:\n" + enum_pipeline_ouputs(generation_gpt2, prompt, 3))

# 处理大型数据集
download_config = DownloadConfig(delete_extracted=True)
dataset = load_dataset("./codeparrot", split="train", download_config=download_config)

print(f"Number of python files code in dataset : {len(dataset)}")
ds_size = sum(os.stat(f["filename"]).st_size for f in dataset.cache_files)
# os.stat.st_size is expressed in bytes, so we convert to GB
print(f"Dataset size (cache file) : {ds_size / 2 ** 30:.2f} GB")
# Process.memory_info is expressed in bytes, so we convert to MB
print(f"RAM used: {psutil.Process(os.getpid()).memory_info().rss >> 20} MB")
streamed_dataset = load_dataset('./codeparrot', split="train", streaming=True)
iterator = iter(streamed_dataset)
print(dataset[0] == next(iterator))
print(dataset[1] == next(iterator))
remote_dataset = load_dataset('transformersbook/codeparrot', split="train", streaming=True)


# 构建一个标记化器


def tok_list(tokenizer, string):
    input_ids = tokenizer(string, add_special_tokens=False)["input_ids"]
    return [tokenizer.decode(tok) for tok in input_ids]


tokenizer_T5 = AutoTokenizer.from_pretrained("t5-base")
tokenizer_camembert = AutoTokenizer.from_pretrained("camembert-base")
print(f'T5 tokens for "sex": {tok_list(tokenizer_T5, "sex")}')
print(f'CamemBERT tokens for "being": {tok_list(tokenizer_camembert, "being")}')

python_code = r"""
def say_hello(): 
	print("Hello, World!") 
# Print it say_hello() """
tokenizer = AutoTokenizer.from_pretrained("gpt2")
print(tokenizer(python_code).tokens())
print(tokenizer.backend_tokenizer.normalizer)
print(tokenizer.backend_tokenizer.pre_tokenizer.pre_tokenize_str(python_code))
a, e = u"a", u"€"
byte = ord(a.encode("utf-8"))
print(f'`{a}` is encoded as `{a.encode("utf-8")}` with a single byte: {byte}')
byte = [ord(chr(i)) for i in e.encode("utf-8")]
print(f'`{e}` is encoded as `{e.encode("utf-8")}` with three bytes: {byte}')

byte_to_unicode_map = byte()
unicode_to_byte_map = dict((v, k) for k, v in byte_to_unicode_map.items())
base_vocab = list(unicode_to_byte_map.keys())
print(f'Size of our base vocabulary: {len(base_vocab)}')
print(f'First element: `{base_vocab[0]}`, last element: `{base_vocab[-1]}`')
print(tokenizer.backend_tokenizer.pre_tokenizer.pre_tokenize_str(python_code))
print(f"Size of the vocabulary: {len(tokenizer)}")
print(tokenizer(python_code).tokens())
tokens = sorted(tokenizer.vocab.items(), key=lambda x: len(x[0]), reverse=True)
print([f'{tokenizer.convert_tokens_to_string(t)}' for t, _ in tokens[:8]])
tokens = sorted(tokenizer.vocab.items(), key=lambda x: x[1], reverse=True)
print([f'{tokenizer.convert_tokens_to_string(t)}' for t, _ in tokens[:12]])

length = 10000
dataset_name = 'transformersbook/codeparrot-train'
dataset = load_dataset(dataset_name, split="train", streaming=True)
iter_dataset = iter(dataset)


def batch_iterator(batch_size=10):
    for _ in tqdm(range(0, length, batch_size)):
        yield [next(iter_dataset)['content'] for _ in range(batch_size)]


new_tokenizer = tokenizer.train_new_from_iterator(batch_iterator(), vocab_size=12500, initial_alphabet=base_vocab)
tokens = sorted(new_tokenizer.vocab.items(), key=lambda x: x[1], reverse=False)
print([f'{tokenizer.convert_tokens_to_string(t)}' for t, _ in tokens[257:280]])
print([f'{new_tokenizer.convert_tokens_to_string(t)}' for t, _ in tokens[-12:]])
print(new_tokenizer(python_code).tokens())
print(f'There are in total {len(keyword.kwlist)} Python keywords.')
for keyw in keyword.kwlist:
    if keyw not in new_tokenizer.vocab:
        print(f'No, keyword `{keyw}` is not in the vocabulary')
length = 200000
new_tokenizer_larger = tokenizer.train_new_from_iterator(batch_iterator(), vocab_size=32768,
                                                         initial_alphabet=base_vocab)
tokens = sorted(new_tokenizer_larger.vocab.items(), key=lambda x: x[1], reverse=False)
print([f'{tokenizer.convert_tokens_to_string(t)}' for t, _ in tokens[-12:]])
print(new_tokenizer_larger(python_code).tokens())
for keyw in keyword.kwlist:
    if keyw not in new_tokenizer_larger.vocab:
        print(f'No, keyword `{keyw}` is not in the vocabulary')
# 在Hub上保存自定义标记器
model_ckpt = "codeparrot"
org = "transformersbook"
new_tokenizer_larger.push_to_hub(model_ckpt, organization=org)
reloaded_tokenizer = AutoTokenizer.from_pretrained(org + "/" + model_ckpt)
print(reloaded_tokenizer(python_code).tokens())
new_tokenizer.push_to_hub(model_ckpt + "-small-vocabulary", organization=org)
# 从头开始训练一个模型
tokenizer = AutoTokenizer.from_pretrained(model_ckpt)
config = AutoConfig.from_pretrained("gpt2-xl", vocab_size=len(tokenizer))
model = AutoModelForCausalLM.from_config(config)
print(f'GPT-2 (xl) size: {model_size(model) / 1000 ** 2:.1f}M parameters')
model.save_pretrained("models/" + model_ckpt, push_to_hub=True, organization=org)
tokenizer = AutoTokenizer.from_pretrained(model_ckpt)
config_small = AutoConfig.from_pretrained("gpt2", vocab_size=len(tokenizer))
model_small = AutoModelForCausalLM.from_config(config_small)
print(f'GPT-2 size: {model_size(model_small) / 1000 ** 2:.1f}M parameters')
model_small.save_pretrained("models/" + model_ckpt + "-small", push_to_hub=True, organization=org)
examples, total_characters, total_tokens = 500, 0, 0
dataset = load_dataset('transformersbook/codeparrot-train', split='train', streaming=True)
for _, example in tqdm(zip(range(examples), iter(dataset)), total=examples):
    total_characters += len(example['content'])
    total_tokens += len(tokenizer(example['content']).tokens())
characters_per_token = total_characters / total_tokens
print(characters_per_token)
shuffled_dataset = dataset.shuffle(buffer_size=100)
constant_length_dataset = ConstantLengthDataset(tokenizer, shuffled_dataset, num_of_sequences=10)
dataset_iterator = iter(constant_length_dataset)
lengths = [len(b) for _, b in zip(range(5), dataset_iterator)]
print(f"Lengths of the sequences: {lengths}")

# Commented parameters correspond to the small model
config = {"train_batch_size": 2,  # 12
          "valid_batch_size": 2,  # 12
          "weight_decay": 0.1, "shuffle_buffer": 1000, "learning_rate": 2e-4,  # 5e-4
          "lr_scheduler_type": "cosine", "num_warmup_steps": 750,  # 2000
          "gradient_accumulation_steps": 16,  # 1 "max_train_steps": 50000, # 150000
          "max_eval_steps": -1, "seq_length": 1024, "seed": 1, "save_checkpoint_steps": 50000}  # 15000
args = Namespace(**config)


def log_metrics(step, metrics):
    logger.info(f"Step {step}: {metrics}"
    if accelerator.is_main_process:
        wandb.log(metrics)[tb_writer.add_scalar(k, v, step) for k, v in metrics.items()]

from transformers import pipeline, set_seed
model_ckpt = 'transformersbook/codeparrot-small'
generation = pipeline('text-generation', model=model_ckpt, device=0)
prompt = '''def area_of_rectangle(a: float, b: float): """Return the area of the rectangle."""'''
complete_code(generation, prompt)
    prompt = '''def get_urls_from_html(html): """Get all embedded URLs in a HTML string."""'''
    complete_code(generation, prompt)
    if not html:
        return []
    return [url for url in re.findall(r'<a href="(/[^/]+/[^"]+?)">',
                                      html)] == == == == == == == == == == == == == == == == == == == == == == == == == == == == == == == == == == == == == == ==
    == return [url for url in re.findall(r'<a href="(.*?)"', html) if
               url] == == == == == == == == == == == == == == == == == == == == == == == == == == == == == == == == == == == == == == ==
    =
    return [url for url in re.findall(r'<a href="(/.*)",',
                                      html)] == == == == == == == == == == == == == == == == == == == == == == == == == == == == == == == == == == == == == == ==
    == return re.findall(r'<a href="(.*?)" class="url"[^>]*>', html)

import requests
def get_urls_from_html(html):
	return [url for url in re.findall(r'<a href="(.*?)"', html) if url]
print(" | ".join(get_urls_from_html(requests.get('https://hf.co/').text)))
model_ckpt = 'transformersbook/codeparrot'
generation = pipeline('text-generation', model=model_ckpt, device=0)
prompt = '''
# a function in native python: 
def mean(a): 
	return sum(a)/len(a) 
# the same function using numpy: import numpy as np def mean(a):'''

complete_code(generation, prompt, max_length=64)
prompt = '''X = np.random.randn(100, 100) 
y = np.random.randint(0, 1, 100) 
# fit random forest classifier with 20 estimators'''
complete_code(generation, prompt, max_length=96)



