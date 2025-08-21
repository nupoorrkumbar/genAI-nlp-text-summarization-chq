# 🧾 NLP-Text-Summarization-chq
### Implementation of LSTM, BiLSTM and GRU+BiLSTM architectures with attention for abstractive text summarization model for consumer health questions  

---

## 📌 Overview  
This project implements **automatic text summarization** for **Consumer Health Questions (CHQs)** using **Bi-directional LSTMs with Attention**.  
The goal is to generate **concise, medically-relevant summaries** that reduce redundancy and help healthcare practitioners and researchers quickly interpret patient queries.  

---

## 🎯 Objectives  
- Apply **NLP techniques** to healthcare domain queries.  
- Build **abstractive summarization models**.  
- Evaluate performance using **ROUGE metrics**.  
- Identify challenges like **hallucination** and **repetition** in generated summaries.  

---

## 🛠️ Tech Stack  
- **Languages/Frameworks:** Python, TensorFlow / Keras  
- **Libraries:** Pandas, NumPy, NLTK, spaCy, Scikit-learn  
- **Deep Learning:** Bi-LSTM, LSTM, GRU+BiLSTM Attention Layer  
- **Evaluation:** ROUGE  

---

# Code Setup Instructions
The entire implementation was developed using Google Colab for ease of access and GPU support. Each model (LSTM, BiLSTM, Hybrid GRU+BiLSTM) is organized in a separate notebook, with a shared preprocessing script (preprocessingword2vec.py) to handle all data cleaning, tokenization, embedding preparation, and padding steps.
The project uses pretrained and frozen Word2Vec embeddings, which were custom-trained on the dataset vocabulary and then frozen during model training to ensure faster convergence and consistent word representations.

⸻

# Setup Steps
	1.	Open the model notebook in Google Colab in each separate tabs:
	•	LSTM.ipynb
	•	BiLSTM.ipynb
	•	GRU_BiLSTM.ipynb
	2.	Upload the following files into the /content/ directory in your Google Colab session:
	a. Dataset files (from the dataset folder):
	•	MeQSum_ACL2019_BenAbacha_Demner-Fushman.xlsx  (Training Set)
	•	MEDIQA2021-Task1-QuestionSummarization-ValidationSet.xlsx  (Validation Set)
	•	MEDIQA2021-Task1-TestSet-ReferenceSummaries.xlsx  (Test Set)
	b. Preprocessing script:
	•	preprocessingword2vec.py
	c. Pretrained embedding model:
	•	w2v_frozen.model
	3.	Switch the Colab runtime to GPU (T4 recommended) under Runtime > Change Runtime Type.
	4.	Run the cells sequentially to train and evaluate the model.

