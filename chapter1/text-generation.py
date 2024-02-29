from transformers import pipeline
import pandas as pd

text = """Dear Amazon, last week I ordered an Optimus Prime action figure from your online store in Germany. 
Unfortunately, when I opened the package, I discovered to my horror that I had been sent an action figure of Megatron 
instead! As a lifelong enemy of the Decepticons, I hope you can understand my dilemma. To resolve the issue, 
I demand an exchange of Megatron for the Optimus Prime figure I ordered. Enclosed are copies of my records concerning 
this purchase. I expect to hear from you soon. Sincerely, Bumblebee."""
generator = pipeline("text-generation")
response = "Dear Bumblebee, I am sorry to hear that your order was mixed up."
prompt = text + "\n\nCustomer service response:\n" + response
outputs = generator(prompt, max_length=200)
print(outputs[0]['generated_text'])
