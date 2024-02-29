from transformers import pipeline
import soundfile as sf
from IPython.display import Audio
from datasets import load_dataset

asr = pipeline("automatic-speech-recognition")
ds = load_dataset("superb", "asr", split="validation[:1]")
print(ds[0])


def map_to_array(batch):
    speech, _ = sf.read(batch["file"])
    batch["speech"] = speech
    return batch


ds = ds.map(map_to_array)
display(Audio(ds[0]['speech'], rate=16000))
pred = asr(ds[0]["speech"])
print(pred)
