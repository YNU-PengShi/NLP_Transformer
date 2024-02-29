from datasets import get_dataset_config_names
from datasets import load_dataset
from collections import defaultdict
from datasets import DatasetDict
import pandas as pd
from collections import Counter

xtreme_subsets = get_dataset_config_names("xtreme")
panx_subsets = [s for s in xtreme_subsets if s.startswith("PAN")]

load_dataset("xtreme", name="PAN-X.de")

langs = ["de", "fr", "it", "en"]
fracs = [0.629, 0.229, 0.084, 0.059]
# Return a DatasetDict if a key doesn't exist
panx_ch = defaultdict(DatasetDict)
for lang, frac in zip(langs, fracs):
    # Load monolingual corpus
    ds = load_dataset("xtreme", name=f"PAN-X.{lang}")
    # Shuffle and downsample each split according to spoken proportion
    for split in ds:
        panx_ch[lang][split] = (ds[split]
                                .shuffle(seed=0)
                                .select(range(int(frac * ds[split].num_rows))))

pd.DataFrame({lang: [panx_ch[lang]["train"].num_rows] for lang in langs}, index=["Number of training examples"])
element = panx_ch["de"]["train"][0]
for key, value in element.items():
    print(f"{key}: {value}")
for key, value in panx_ch["de"]["train"].features.items():
    print(f"{key}: {value}")
tags = panx_ch["de"]["train"].features["ner_tags"].feature
print(tags)


def create_tag_names(batch):
    return {"ner_tags_str": [tags.int2str(idx) for idx in batch["ner_tags"]]}


panx_de = panx_ch["de"].map(create_tag_names)
de_example = panx_de["train"][0]
pd.DataFrame([de_example["tokens"], de_example["ner_tags_str"]], ['Tokens', 'Tags'])

split2freqs = defaultdict(Counter)
for split, dataset in panx_de.items():
    for row in dataset["ner_tags_str"]:
        for tag in row:
            if tag.startswith("B"):
                tag_type = tag.split("-")[1]
                split2freqs[split][tag_type] += 1

pd.DataFrame.from_dict(split2freqs, orient="index")

