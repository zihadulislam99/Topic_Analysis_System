from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

MODEL_PATH = "./Topic_Analysis_System/topic_model"
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, local_files_only=True)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH, local_files_only=True)
model.eval()  # IMPORTANT for inference

def predict_topic(texts):
    if isinstance(texts, str):
        texts = [texts]
    inputs = tokenizer(texts, return_tensors="pt", truncation=True, padding=True, max_length=512)

    with torch.no_grad():
        outputs = model(**inputs)
    probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
    topic_map = {0: "Military", 1: "Border", 2: "Terrorism", 3: "Government", 4: "Elections", 5: "Law enforcement", 6: "Economy", 7: "Other"}
    # topic_map = {0: "Military", 1: "Border", 2: "Terrorism", 3: "Government", 4: "Elections", 5: "Law enforcement", 6: "Economy", 7: "Other"}
    return [topic_map[p] for p in torch.argmax(probs, dim=-1).tolist()]

texts = [ "I will kill you", "The government must be destroyed", "Banks will collapse tomorrow", "Drinking bleach cures disease", "I love learning AI", "সারা দেশে ব্যাংক বন্ধ হয়ে যাবে", "Ti amo", "I love you"]

print(predict_topic("Border security forces increased patrols.")[0])
# print(predict_topic(["Ti amo", "I love you"]))
# for text, topic in zip(texts, predict_topic(texts)):
#     print(f"Text: {text}  ||  topic: {topic}")
