# from datasets import list_datasets
from datasets import load_dataset
import matplotlib.pyplot as plt
import pandas as pd
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel
from transformers import DistilBertTokenizer
from sklearn.linear_model import LogisticRegression
from sklearn.dummy import DummyClassifier
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix
from transformers import AutoModelForSequenceClassification
from sklearn.metrics import accuracy_score, f1_score

# all_datasets = list_datasets()
emotions = load_dataset("emotion")
train_ds = emotions["train"]
# print(train_ds)
# print(len(train_ds))
# print(train_ds[0])
# print(train_ds.column_names)
# print(train_ds.features)
# print(train_ds[:5])
# print(train_ds["text"][:5])

emotions.set_format(type="pandas")
df = emotions["train"][:]


def label_int2str(row):
    return emotions["train"].features["label"].int2str(row)


df["label_name"] = df["label"].apply(label_int2str)
print(df.head())

# 查看类分布
df["label_name"].value_counts(ascending=True).plot.barh()
plt.title("Frequency of Classes")
plt.show()

# 查看推文的长度
df["Words Per Tweet"] = df["text"].str.split().apply(len)
df.boxplot("Words Per Tweet", by="label_name", grid=False, showfliers=False, color="black")
plt.suptitle("")
plt.xlabel("")
plt.show()

text = "Tokenizing text is a core task of NLP."
# 字符标记化
tokenized_text = list(text)
token2idx = {ch: idx for idx, ch in enumerate(sorted(set(tokenized_text)))}
input_ids = [token2idx[token] for token in tokenized_text]

categorical_df = pd.DataFrame({"Name": ["Bumblebee", "Optimus Prime", "Megatron"], "Label ID": [0, 1, 2]})
# 生成独热向量
pd.get_dummies(categorical_df["Name"])
input_ids = torch.tensor(input_ids)
one_hot_encodings = F.one_hot(input_ids, num_classes=len(token2idx))
print(one_hot_encodings.shape)

# 词标记化
tokenized_text = text.split()

model_ckpt = "distilbert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_ckpt)
distilbert_tokenizer = DistilBertTokenizer.from_pretrained(model_ckpt)
encoded_text = tokenizer(text)
tokens = tokenizer.convert_ids_to_tokens(encoded_text.input_ids)

print(tokenizer.convert_tokens_to_string(tokens))


# 检查词汇量的大小：tokenizer.vocab_size
# 相应模型的最大上下文大小：tokenizer.model_max_length
# 模型在其前向传递中期望的字段的名称：tokenizer.model_input_names


# 对整个数据集进行标记化
def tokenize(batch):
    return tokenizer(batch["text"], padding=True, truncation=True)


print(tokenize(emotions["train"][:2]))
emotions_encoded = emotions.map(tokenize, batched=True, batch_size=None)
print(emotions_encoded["train"].column_names)

# 判定使用CPU还说gpu
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = AutoModel.from_pretrained(model_ckpt).to(device)

text = "this is a test"
inputs = tokenizer(text, return_tensors="pt")
print(f"Input tensor shape: {inputs['input_ids'].size()}")
inputs = {k: v.to(device)
          for k, v in inputs.items()}
with torch.no_grad():
    outputs = model(**inputs)
# 最后一个隐藏状态
outputs.last_hidden_state.size()
# 由于这个标记出现在每个序列的开始，我们可以通过简单地索引到output.last_hidden_state来提取它
outputs.last_hidden_state[:, 0].size()


# 创建一个新的hidden_state列
def extract_hidden_states(batch):  # Place model inputs on the GPU
    inputs = {
        k: v.to(device) for k, v in batch.items() if k in tokenizer.model_input_names
    }  # Extract last hidden states
    with torch.no_grad():
        last_hidden_state = model(**inputs).last_hidden_state  # Return vector for [CLS] token
    return {"hidden_state": last_hidden_state[:, 0].cpu().numpy()}


emotions_encoded.set_format("torch", columns=["input_ids", "attention_mask", "label"])
emotions_hidden = emotions_encoded.map(extract_hidden_states, batched=True)
print(emotions_hidden["train"].column_names)

# 创建特征矩阵
import numpy as np
from umap import UMAP
from sklearn.preprocessing import MinMaxScaler  # Scale features to [0,1] range

X_train = np.array(emotions_hidden["train"]["hidden_state"])
X_valid = np.array(emotions_hidden["validation"]["hidden_state"])
y_train = np.array(emotions_hidden["train"]["label"])
y_valid = np.array(emotions_hidden["validation"]["label"])
X_scaled = MinMaxScaler().fit_transform(X_train)
# Initialize and fit UMAP
mapper = UMAP(n_components=2, metric="cosine").fit(X_scaled)
# Create a DataFrame of 2D embeddings
df_emb = pd.DataFrame(mapper.embedding_, columns=["X", "Y"])
df_emb["label"] = y_train
df_emb.head()
# 绘制每一类的点的密度
fig, axes = plt.subplots(2, 3, figsize=(7, 5))
axes = axes.flatten()
cmaps = ["Greys", "Blues", "Oranges", "Reds", "Purples", "Greens"]
labels = emotions["train"].features["label"].names
for i, (label, cmap) in enumerate(zip(labels, cmaps)):
    df_emb_sub = df_emb.query(f"label == {i}")
    axes[i].hexbin(df_emb_sub["X"], df_emb_sub["Y"], cmap=cmap, gridsize=20, linewidths=(0,))
    axes[i].set_title(label)
axes[i].set_xticks([]), axes[i].set_yticks([])
plt.tight_layout()
plt.show()

# 训练一个简单的分类器

# We increase `max_iter` to guarantee convergence
lr_clf = LogisticRegression(max_iter=3000)
lr_clf.fit(X_train, y_train)
lr_clf.score(X_valid, y_valid)

dummy_clf = DummyClassifier(strategy="most_frequent")
dummy_clf.fit(X_train, y_train)
dummy_clf.score(X_valid, y_valid)


def plot_confusion_matrix(y_preds, y_true, labels):
    cm = confusion_matrix(y_true, y_preds, normalize="true")
    fig, ax = plt.subplots(figsize=(6, 6))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
    disp.plot(cmap="Blues", values_format=".2f", ax=ax, colorbar=False)
    plt.title("Normalized confusion matrix")
    plt.show()


y_preds = lr_clf.predict(X_valid)
plot_confusion_matrix(y_preds, y_valid, labels)

# 微调Transformers
num_labels = 6
model = (AutoModelForSequenceClassification.from_pretrained(model_ckpt, num_labels=num_labels).to(device))


# 定义性能指标
def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    f1 = f1_score(labels, preds, average="weighted")
    acc = accuracy_score(labels, preds)
    return {"accuracy": acc, "f1": f1}


# 训练模型
from huggingface_hub import notebook_login

notebook_login()
from transformers import Trainer, TrainingArguments

batch_size = 64
logging_steps = len(emotions_encoded["train"]) // batch_size
model_name = f"{model_ckpt}-finetuned-emotion"
training_args = TrainingArguments(output_dir=model_name, num_train_epochs=2, learning_rate=2e-5,
                                  per_device_train_batch_size=batch_size, per_device_eval_batch_size=batch_size,
                                  weight_decay=0.01, evaluation_strategy="epoch", disable_tqdm=False,
                                  logging_steps=logging_steps, push_to_hub=True, log_level="error")
trainer = Trainer(model=model, args=training_args, compute_metrics=compute_metrics,
                  train_dataset=emotions_encoded["train"], eval_dataset=emotions_encoded["validation"],
                  tokenizer=tokenizer)
trainer.train()
preds_output = trainer.predict(emotions_encoded["validation"])
print(preds_output.metrics)
y_preds = np.argmax(preds_output.predictions, axis=1)
# 画混淆矩阵
plot_confusion_matrix(y_preds, y_valid, labels)

# 错误分析
from torch.nn.functional import cross_entropy


def forward_pass_with_label(batch):
    inputs = {k: v.to(device) for k, v in batch.items() if k in tokenizer.model_input_names}
    with torch.no_grad(): output = model(**inputs)
    pred_label = torch.argmax(output.logits, axis=-1)
    loss = cross_entropy(output.logits, batch["label"].to(device), reduction="none")
    # Place outputs on CPU for compatibility with other dataset columns
    return {"loss": loss.cpu().numpy(), "predicted_label": pred_label.cpu().numpy()}


# Convert our dataset back to PyTorch tensors
emotions_encoded.set_format("torch", columns=["input_ids", "attention_mask", "label"])  # Compute loss values
emotions_encoded["validation"] = emotions_encoded["validation"].map(forward_pass_with_label, batched=True,
                                                                    batch_size=16)
emotions_encoded.set_format("pandas")
cols = ["text", "label", "predicted_label", "loss"]
df_test = emotions_encoded["validation"][:][cols]
df_test["label"] = df_test["label"].apply(label_int2str)
df_test["predicted_label"] = (df_test["predicted_label"].apply(label_int2str))
df_test.sort_values("loss", ascending=False).head(10)
