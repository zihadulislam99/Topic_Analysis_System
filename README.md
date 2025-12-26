[![MIT License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-Deep%20Learning-red.svg)](https://pytorch.org/)
[![Transformers](https://img.shields.io/badge/Hugging%20Face-Transformers-yellow.svg)](https://huggingface.co/transformers/)
[![NLP](https://img.shields.io/badge/NLP-Topic%20Analysis-purple.svg)](#)
[![Status](https://img.shields.io/badge/Status-Active-success.svg)](CONTRIBUTING.md)
[![Contributions Welcome](https://img.shields.io/badge/Contributions-Welcome-brightgreen.svg)](CONTRIBUTING.md)

# Multilingual Topic Analysis System

A **multilingual topic classification system** built using **Python, PyTorch, and Hugging Face Transformers**.  
The system classifies text into **eight sensitive topic categories** such as *Military, Border, Terrorism, Government, Elections, Law Enforcement, Economy,* and *Other*.

It supports **English and Bengali (বাংলা)** and is designed to work **fully offline**, making it suitable for **secure, air-gapped, or low-connectivity environments**.

---

## **Features**

* **Multilingual Topic Classification**
* **8-Class Topic Categorization**
* **English + Bengali (Bangla) Support**
* **Offline Inference (Local Model Loading)**
* **Batch & Single Text Processing**
* **Transformer-Based Deep Learning Model**
* **Lightweight Inference Mode**
* **Easy Integration with APIs & Pipelines**

---

## **Topic Labels**

| Label ID | Topic Category     |
|--------:|-------------------|
| 0       | Military          |
| 1       | Border            |
| 2       | Terrorism         |
| 3       | Government        |
| 4       | Elections         |
| 5       | Law Enforcement   |
| 6       | Economy           |
| 7       | Other             |

---

## **Task Details**

| Property              | Description                                  |
|----------------------|----------------------------------------------|
| **Task**             | Text Classification (Topic Analysis)         |
| **Number of Classes**| 8                                            |
| **Languages**        | English, Bengali (বাংলা)                     |
| **Framework**        | PyTorch                                     |
| **Model Type**       | Transformer (Hugging Face)                  |
| **Inference Mode**   | Offline / Local                             |

---

## **Technology Stack**

* **Programming Language:** Python  
* **Deep Learning Framework:** PyTorch  
* **NLP Library:** Hugging Face Transformers  
* **Tokenizer:** AutoTokenizer  
* **Model Loader:** AutoModelForSequenceClassification  

---

## **Project Structure**

```

Topic_Analysis_System/
│
├── topic_model/              # Local trained model & tokenizer
│   ├── config.json
│   ├── pytorch_model.bin
│   ├── tokenizer.json
│   └── vocab files
│
├── inference.py               # Topic prediction script
├── requirements.txt           # Dependencies
└── README.md                  # Documentation

````

---

## **Setup Instructions**

### 1. Install Dependencies

```bash
pip install torch transformers datasets
````

> ⚠️ Internet is **NOT required during inference** if the model files are stored locally.

---

### 2. Model Preparation

Ensure your trained model exists locally at:

```
./topic_model
```

Model loading is done using:

```python
AutoModelForSequenceClassification.from_pretrained(
    MODEL_PATH,
    local_files_only=True
)
```

---

### 3. Run Topic Prediction

Example inference code:

```python
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

MODEL_PATH = "./topic_model"

tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, local_files_only=True)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH, local_files_only=True)
model.eval()

label_map = {
    0: "Military",
    1: "Border",
    2: "Terrorism",
    3: "Government",
    4: "Elections",
    5: "Law Enforcement",
    6: "Economy",
    7: "Other"
}

def predict_topic(texts):
    if isinstance(texts, str):
        texts = [texts]

    inputs = tokenizer(
        texts,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=512
    )

    with torch.no_grad():
        outputs = model(**inputs)
        predictions = torch.argmax(outputs.logits, dim=-1)

    return [label_map[p.item()] for p in predictions]
```

---

## **Usage Example**

```python
print(predict_topic("সীমান্তে নতুন সামরিক মোতায়েন করা হয়েছে")[0])
# Output: Military

texts = [
    "The government announced a new policy today",
    "Election campaigns are intensifying"
]
print(predict_topic(texts))
# Output: ['Government', 'Elections']
```

---

## **Best Practices**

* Keep text length under **512 tokens**
* Use complete, clear sentences
* Batch processing improves performance
* Suitable for news, social media, and intelligence data

---

## **Applications**

* News topic classification
* OSINT and media monitoring
* Risk and sensitivity analysis
* Policy and governance analytics
* Multilingual NLP systems
* Offline intelligence systems

---

## **License**

This project is licensed under the **MIT License** — free for **personal, educational, and research use**.

---

## **Author**

**Zihadul Islam**
GitHub: [https://github.com/zihadulislam99](https://github.com/zihadulislam99)
