import pandas as pd
from transformers import pipeline

book_data = [
    {"chapter": 0, "name": "Introduction", "start_page": 1, "end_page": 11},
    {"chapter": 1, "name": "Text classification", "start_page": 12, "end_page": 48},
    {"chapter": 2, "name": "Named Entity Recognition", "start_page": 49, "end_page": 73},
    {"chapter": 3, "name": "Question Answering", "start_page": 74, "end_page": 120},
    {"chapter": 4, "name": "Summarization", "start_page": 121, "end_page": 140},
    {"chapter": 5, "name": "Conclusion", "start_page": 141, "end_page": 144}
]
table = pd.DataFrame(book_data)
table['number_of_pages'] = table['end_page'] - table['start_page']
table = table.astype(str)
print(table)
table_qa = pipeline("table-question-answering")
queries = ["What's the topic in chapter 4?", "What is the total number of pages?",
           "On which page does the chapter about question-answering start?",
           "How many chapters have more than 20 pages?"]
preds = table_qa(table, queries)
table_qa = pipeline("table-question-answering")
for query, pred in zip(queries, preds):
    print(query)
    if pred["aggregator"] == "NONE":
        print("Predicted answer: " + pred["answer"])
    else:
        print("Predicted answer: " + pred["answer"])
    print('=' * 50)
