from datasets import get_dataset_config_names
from datasets import load_dataset
import pandas as pd
from matplotlib import pyplot as plt
from transformers import AutoTokenizer, pipeline
import torch
from transformers import AutoModelForQuestionAnswering
from farm.evaluation.squad_evaluation import compute_f1, compute_exact
import json
from haystack.generator.transformers import RAGenerator
from haystack.pipeline import GenerativeQAPipeline

# 数据集
domains = get_dataset_config_names("subjqa")
subjqa = load_dataset("subjqa", name="electronics")
print(subjqa["train"]["answers"][1])
dfs = {split: dset.to_pandas() for split, dset in subjqa.flatten().items()}
for split, df in dfs.items():
    print(f"Number of questions in {split}: {df['id'].nunique()}")
qa_cols = ["title", "question", "answers.text", "answers.answer_start", "context"]
sample_df = dfs["train"][qa_cols].sample(2, random_state=7)
start_idx = sample_df["answers.answer_start"].iloc[0][0]
end_idx = start_idx + len(sample_df["answers.text"].iloc[0][0])
print(sample_df["context"].iloc[0][start_idx:end_idx])
counts = {}
question_types = ["What", "How", "Is", "Does", "Do", "Was", "Where", "Why"]
for q in question_types:
    counts[q] = dfs["train"]["question"].str.startswith(q).value_counts()
pd.Series(counts).sort_values().plot.barh()
plt.title("Frequency of Question Types")
plt.show()
for question_type in ["How", "What", "Is"]:
    for question in (
            dfs["train"][dfs["train"].question.str.startswith(question_type)].sample(n=3, random_state=42)['question']):
        print(question)

model_ckpt = "deepset/minilm-uncased-squad2"
tokenizer = AutoTokenizer.from_pretrained(model_ckpt)
uestion = "How much music can this hold?"
context = """An MP3 is about 1 MB/minute, so about 6000 hours depending on \ file size."""
inputs = tokenizer(question, context, return_tensors="pt")
print(tokenizer.decode(inputs["input_ids"][0]))
model = AutoModelForQuestionAnswering.from_pretrained(model_ckpt)
with torch.no_grad():
    outputs = model(**inputs)
    print(outputs)
start_logits = outputs.start_logits
end_logits = outputs.end_logits
print(f"Input IDs shape: {inputs.input_ids.size()}")
print(f"Start logits shape: {start_logits.size()}")
print(f"End logits shape: {end_logits.size()}")
start_idx = torch.argmax(start_logits)
end_idx = torch.argmax(end_logits) + 1
answer_span = inputs["input_ids"][0][start_idx:end_idx]
answer = tokenizer.decode(answer_span)
print(f"Question: {question}")
print(f"Answer: {answer}")
pipe = pipeline("question-answering", model=model, tokenizer=tokenizer)
pipe(question=question, context=context, topk=3)
pipe(question="Why is there no data?", context=context, handle_impossible_answer=True)

# 处理长文本
example = dfs["train"].iloc[0][["question", "context"]]
tokenized_example = tokenizer(example["question"], example["context"], return_overflowing_tokens=True, max_length=100,
                              stride=25)
for idx, window in enumerate(tokenized_example["input_ids"]):
    print(f"Window #{idx} has {len(window)} tokens")
for window in tokenized_example["input_ids"]:
    print(f"{tokenizer.decode(window)} \n")

# 评估阅读器
pred = "about 6000 hours"
label = "6000 hours"
print(f"EM: {compute_exact(label, pred)}")
print(f"F1: {compute_f1(label, pred)}")
pred = "about 6000 dollars"
print(f"EM: {compute_exact(label, pred)}")
print(f"F1: {compute_f1(label, pred)}")
from haystack.eval import EvalAnswers


def evaluate_reader(reader):
    score_keys = ['top_1_em', 'top_1_f1']
    eval_reader = EvalAnswers(skip_incorrect_retrieval=False)
    pipe = Pipeline()
    pipe.add_node(component=reader, name="QAReader", inputs=["Query"])
    pipe.add_node(component=eval_reader, name="EvalReader", inputs=["QAReader"])
    for l in labels_agg:
        doc = document_store.query(l.question, filters={"question_id": [l.origin]})
        _ = pipe.run(query=l.question, documents=doc, labels=l)
    return {k: v for k, v in eval_reader.__dict__.items() if k in score_keys}


reader_eval = {}
reader_eval["Fine-tune on SQuAD"] = evaluate_reader(reader)


def plot_reader_eval(reader_eval):
    fig, ax = plt.subplots()
    df = pd.DataFrame.from_dict(reader_eval)
    df.plot(kind="bar", ylabel="Score", rot=0, ax=ax)
    ax.set_xticklabels(["EM", "F1"])
    plt.legend(loc='upper left')
    plt.show()


plot_reader_eval(reader_eval)


def create_paragraphs(df):
    paragraphs = []
    id2context = dict(zip(df["review_id"], df["context"]))
    for review_id, review in id2context.items():
        qas = []
        # Filter for all question-answer pairs about a specific context
        review_df = df.query(f"review_id == '{review_id}'")
        id2question = dict(zip(review_df["id"], review_df["question"]))
        # Build up the qas array
        for qid, question in id2question.items():
        # Filter for a single question ID
        question_df = df.query(f"id == '{qid}'").to_dict(orient="list")
        ans_start_idxs = question_df["answers.answer_start"][0].tolist()
    ans_text = question_df["answers.text"][0].tolist()
    # Fill answerable questions
    if len(ans_start_idxs):
        answers = [{"text": text, "answer_start": answer_start} for text, answer_start in zip(ans_text, ans_start_idxs)]
        is_impossible = False
    else:
        answers = []
        is_impossible = True
    # Add question-answer pairs to
    qas
    qas.append({"question": question, "id": qid, "is_impossible": is_impossible, "answers": answers})
    # Add context and question-answer pairs to paragraphs
    paragraphs.append({"qas": qas, "context": review})
    return paragraphs


product = dfs["train"].query("title == 'B00001P4ZH'")
create_paragraphs(product)


def convert_to_squad(dfs):
    for split, df in dfs.items():
        subjqa_data = {}
        # Create `paragraphs` for each product ID
        groups = (df.groupby("title").apply(create_paragraphs).to_frame(name="paragraphs").reset_index())
        subjqa_data["data"] = groups.to_dict(orient="records")
        # Save the result to disk
        with open(f"electronics-{split}.json", "w+", encoding="utf-8") as f:
            json.dump(subjqa_data, f)


convert_to_squad(dfs)
train_filename = "electronics-train.json"
dev_filename = "electronics-validation.json"
reader.train(data_dir=".", use_gpu=True, n_epochs=1, batch_size=16, train_filename=train_filename,
             dev_filename=dev_filename)
minilm_ckpt = "microsoft/MiniLM-L12-H384-uncased"
minilm_reader = FARMReader(model_name_or_path=minilm_ckpt, progress_bar=False, max_seq_len=max_seq_length,
                           doc_stride=doc_stride, return_no_answer=True)
minilm_reader.train(data_dir=".", use_gpu=True, n_epochs=1, batch_size=16, train_filename=train_filename,
                    dev_filename=dev_filename)
reader_eval["Fine-tune on SubjQA"] = evaluate_reader(minilm_reader)

plot_reader_eval(reader_eval)
pipe = EvalRetrieverPipeline(es_retriever)
eval_reader = EvalAnswers()
pipe.pipeline.add_node(component=reader, name="QAReader", inputs=["EvalRetriever"])
pipe.pipeline.add_node(component=eval_reader, name="EvalReader", inputs=["QAReader"])
run_pipeline(pipe)
reader_eval["QA Pipeline (top-1)"] = {k: v for k, v in eval_reader.__dict__.items() if k in ["top_1_em", "top_1_f1"]}

# 超越抽取式QA

generator = RAGenerator(model_name_or_path="facebook/rag-token-nq", embed_title=False, num_beams=5)

pipe = GenerativeQAPipeline(generator=generator, retriever=dpr_retriever)


def generate_answers(query, top_k_generator=3):
    preds = pipe.run(query=query, top_k_generator=top_k_generator, top_k_retriever=5,
                     filters={"item_id": ["B0074BW614"]})
    print(f"Question: {preds['query']} \n")
    for idx in range(top_k_generator):
        print(f"Answer {idx + 1}: {preds['answers'][idx]['answer']}")


generate_answers(query)
generate_answers("What is the main drawback?")
