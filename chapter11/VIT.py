from PIL import Image
import matplotlib.pyplot as plt
import pandas as pd
from transformers import pipeline

image = Image.open("image/dog.jpg")
plt.imshow(image)
plt.axis("off")
plt.show()
image_classifier = pipeline("image-classification")
preds = image_classifier(image)
preds_df = pd.DataFrame(preds)
print(preds_df)
