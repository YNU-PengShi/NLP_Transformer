from matplotlib import pyplot as plt
from transformers import pipeline
from datasets import load_dataset, load_metric
import torch
from pathlib import Path
from time import perf_counter
from transformers import TrainingArguments, AutoConfig, AutoTokenizer
import numpy as np
from transformers import AutoModelForSequenceClassification
import optuna
from torch import quantize_per_tensor, nn
from torch.nn.quantized import QFunctional
import sys
from torch.quantization import quantize_dynamic
from transformers.convert_graph_to_onnx import convert
import os
from psutil import cpu_count
from onnxruntime import (GraphOptimizationLevel, InferenceSession, SessionOptions)
from scipy.special import softmax
from onnxruntime.quantization import quantize_dynamic, QuantType

bert_ckpt = "transformersbook/bert-base-uncased-finetuned-clinc"
pipe = pipeline("text-classification", model=bert_ckpt)
query = """Hey, I'd like to rent a vehicle from Nov 1st to Nov 15th in Paris and I need a 15 passenger van"""
print(pipe(query))

# 创建一个性能基准
clinc = load_dataset("clinc_oos", "plus")
sample = clinc["test"][42]
print(sample)
intents = clinc["test"].features["intent"]
intents.int2str(sample["intent"])
accuracy_score = load_metric("accuracy")


def compute_accuracy(self):
    """This overrides the PerformanceBenchmark.compute_accuracy() method"""
    preds, labels = [], []
    for example in self.dataset:
        pred = self.pipeline(example["text"])[0]["label"]
        label = example["intent"]
        preds.append(intents.str2int(pred))
        labels.append(label)
    accuracy = accuracy_score.compute(predictions=preds, references=labels)
    print(f"Accuracy on test set - {accuracy['accuracy']:.3f}")
    return accuracy
    PerformanceBenchmark.compute_accuracy = compute_accuracy


print(list(pipe.model.state_dict().items())[42])
torch.save(pipe.model.state_dict(), "model.pt")


def compute_size(self):
    """This overrides the PerformanceBenchmark.compute_size() method"""
    state_dict = self.pipeline.model.state_dict()
    tmp_path = Path("model.pt")
    torch.save(state_dict, tmp_path)
    # Calculate size in megabytes
    size_mb = Path(tmp_path).stat().st_size / (1024 * 1024)
    # Delete temporary file
    tmp_path.unlink()
    print(f"Model size (MB) - {size_mb:.2f}")
    return {"size_mb": size_mb}


for _ in range(3):
    start_time = perf_counter()
    _ = pipe(query)
    latency = perf_counter() - start_time
    print(f"Latency (ms) - {1000 * latency:.3f}")
pb = PerformanceBenchmark(pipe, clinc["test"])
perf_metrics = pb.run_benchmark()


# 知识蒸馏
class DistillationTrainingArguments(TrainingArguments):
    def __init__(self, *args, alpha=0.5, temperature=2.0, **kwargs):
        super().__init__(*args, **kwargs)
        self.alpha = alpha
        self.temperature = temperature


def compute_metrics(pred):
    predictions, labels = pred
    predictions = np.argmax(predictions, axis=1)
    return accuracy_score.compute(predictions=predictions, references=labels)


batch_size = 48
finetuned_ckpt = "distilbert-base-uncased-finetuned-clinc"
student_training_args = DistillationTrainingArguments(output_dir=finetuned_ckpt, evaluation_strategy="epoch",
                                                      num_train_epochs=5, learning_rate=2e-5,
                                                      per_device_train_batch_size=batch_size,
                                                      per_device_eval_batch_size=batch_size, alpha=1, weight_decay=0.01,
                                                      push_to_hub=True)
id2label = pipe.model.config.id2label
label2id = pipe.model.config.label2id
num_labels = intents.num_classes
student_ckpt = "distilbert-base-uncased"
student_tokenizer = AutoTokenizer.from_pretrained(student_ckpt)
student_config = (AutoConfig.from_pretrained(student_ckpt, num_labels=num_labels, id2label=id2label, label2id=label2id))
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def student_init():
    return AutoModelForSequenceClassification.from_pretrained(student_ckpt, config=student_config).to(device)


teacher_ckpt = "transformersbook/bert-base-uncased-finetuned-clinc"
teacher_model = (AutoModelForSequenceClassification.from_pretrained(teacher_ckpt, num_labels=num_labels).to(device))
distilbert_trainer = DistillationTrainer(model_init=student_init, teacher_model=teacher_model,
                                         args=student_training_args, train_dataset=clinc_enc['train'],
                                         eval_dataset=clinc_enc['validation'], compute_metrics=compute_metrics,
                                         tokenizer=student_tokenizer)

distilbert_trainer.train()
distilbert_trainer.push_to_hub("Training completed!")
finetuned_ckpt = "transformersbook/distilbert-base-uncased-finetuned-clinc"
pipe = pipeline("text-classification", model=finetuned_ckpt)
optim_type = "DistilBERT"
pb = PerformanceBenchmark(pipe, clinc["test"], optim_type=optim_type)
perf_metrics.update(pb.run_benchmark())


# 用Optuna寻找好的超参数
def objective(trial):
    x = trial.suggest_float("x", -2, 2)
    y = trial.suggest_float("y", -2, 2)
    return (1 - x) ** 2 + 100 * (y - x ** 2) ** 2


study = optuna.create_study()
study.optimize(objective, n_trials=1000)
print(study.best_params)


def hp_space(trial):
    return {"num_train_epochs": trial.suggest_int("num_train_epochs", 5, 10),
            "alpha": trial.suggest_float("alpha", 0, 1), "temperature": trial.suggest_int("temperature", 2, 20)}


best_run = distilbert_trainer.hyperparameter_search(n_trials=20, direction="maximize", hp_space=hp_space)
print(best_run)
distil_trainer.push_to_hub("Training complete")
distilled_ckpt = "transformersbook/distilbert-base-uncased-distilled-clinc"
pipe = pipeline("text-classification", model=distilled_ckpt)
optim_type = "Distillation"
pb = PerformanceBenchmark(pipe, clinc["test"], optim_type=optim_type)
perf_metrics.update(pb.run_benchmark())
plot_metrics(perf_metrics, optim_type)

# 用量化技术使模型更快
state_dict = pipe.model.state_dict()
weights = state_dict["distilbert.transformer.layer.0.attention.out_lin.weight"]
plt.hist(weights.flatten().numpy(), bins=250, range=(-0.3, 0.3), edgecolor="C0")
plt.show()
zero_point = 0
scale = (weights.max() - weights.min()) / (127 - (-128))
(weights / scale + zero_point).clamp(-128, 127).round().char()

dtype = torch.qint8
quantized_weights = quantize_per_tensor(weights, scale, zero_point, dtype)
quantized_weights.int_repr()
q_fn = QFunctional()

sys.getsizeof(weights.storage()) / sys.getsizeof(quantized_weights.storage())

model_ckpt = "transformersbook/distilbert-base-uncased-distilled-clinc"
tokenizer = AutoTokenizer.from_pretrained(model_ckpt)
model = (AutoModelForSequenceClassification.from_pretrained(model_ckpt).to("cpu"))
model_quantized = quantize_dynamic(model, {nn.Linear}, dtype=torch.qint8)
pipe = pipeline("text-classification", model=model_quantized, tokenizer=tokenizer)
optim_type = "Distillation + quantization"
pb = PerformanceBenchmark(pipe, clinc["test"], optim_type=optim_type)
perf_metrics.update(pb.run_benchmark())
plot_metrics(perf_metrics, optim_type)

# 用ONNX和ONNXRuntime优化推理


os.environ["OMP_NUM_THREADS"] = f"{cpu_count()}"
os.environ["OMP_WAIT_POLICY"] = "ACTIVE"
model_ckpt = "transformersbook/distilbert-base-uncased-distilled-clinc"
onnx_model_path = Path("onnx/model.onnx")
convert(framework="pt", model=model_ckpt, tokenizer=tokenizer, output=onnx_model_path, opset=12,
        pipeline_name="text-classification")


def create_model_for_provider(model_path, provider="CPUExecutionProvider"):
    options = SessionOptions()
    options.intra_op_num_threads = 1
    options.graph_optimization_level = GraphOptimizationLevel.ORT_ENABLE_ALL
    session = InferenceSession(str(model_path), options, providers=[provider])
    session.disable_fallback()
    return session


onnx_model = create_model_for_provider(onnx_model_path)
inputs = clinc_enc["test"][:1]
del inputs["labels"]
ogits_onnx = onnx_model.run(None, inputs)[0]
print(logits_onnx.shape)
np.argmax(logits_onnx)
clinc_enc["test"][0]["labels"]


class OnnxPipeline:
    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer

    def __call__(self, query):
        model_inputs = self.tokenizer(query, return_tensors="pt")
        inputs_onnx = {k: v.cpu().detach().numpy() for k, v in model_inputs.items()}
        logits = self.model.run(None, inputs_onnx)[0][0, :]
        probs = softmax(logits)
        pred_idx = np.argmax(probs).item()
        return [{"label": intents.int2str(pred_idx), "score": probs[pred_idx]}]


pipe = OnnxPipeline(onnx_model, tokenizer)
pipe(query)


class OnnxPerformanceBenchmark(PerformanceBenchmark):
    def __init__(self, *args, model_path, **kwargs):
        super().__init__(*args, **kwargs)
        self.model_path = model_path

    def compute_size(self):
        size_mb = Path(self.model_path).stat().st_size / (1024 * 1024)
        print(f"Model size (MB) - {size_mb:.2f}")
        return {"size_mb": size_mb}


optim_type = "Distillation + ORT"
pb = OnnxPerformanceBenchmark(pipe, clinc["test"], optim_type, model_path="onnx/model.onnx")
perf_metrics.update(pb.run_benchmark())
plot_metrics(perf_metrics, optim_type)

model_input = "onnx/model.onnx"
model_output = "onnx/model.quant.onnx"
quantize_dynamic(model_input, model_output, weight_type=QuantType.QInt8)
nnx_quantized_model = create_model_for_provider(model_output)
pipe = OnnxPipeline(onnx_quantized_model, tokenizer)
optim_type = "Distillation + ORT (quantized)"
pb = OnnxPerformanceBenchmark(pipe, clinc["test"], optim_type, model_path=model_output)
perf_metrics.update(pb.run_benchmark()
plot_metrics(perf_metrics, optim_type)
