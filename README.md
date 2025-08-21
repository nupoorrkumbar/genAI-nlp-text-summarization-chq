# NLP-Text-Summarization-CHQ 
### Generative AI for Text Summarization of Consumer Health Questions (CHQ) using LSTM, BiLSTM and GRU+BiLSTM architectures with attention for abstractive text summarization model

---

## Overview  
This project applies **Generative AI (GenAI)** techniques to **automatic text summarization** of **Consumer Health Questions (CHQs)**.  
Using **Bi-directional LSTMs with Attention**, the model generates **concise, medically-relevant summaries** that reduce redundancy and help healthcare practitioners and researchers quickly interpret patient queries.  

---

## Objectives  
- Apply **GenAI-based NLP techniques** to healthcare queries.  
- Build **abstractive summarization models** for CHQs.  
- Evaluate model performance using **ROUGE metrics**.  
- Address challenges such as **hallucination** and **repetition**.  

---

## Tech Stack  
- **Languages/Frameworks:** Python, TensorFlow / Keras  
- **Libraries:** Pandas, NumPy, NLTK, spaCy, Scikit-learn  
- **Deep Learning / GenAI:** Bi-LSTM, Attention Layer  
- **Evaluation:** ROUGE  
---

## Key Contributions
- **Preprocessing:** Cleaned and tokenized CHQs with lemmatization and stopword removal.
- **Modeling:** Built a GenAI-driven Bi-LSTM with Attention for abstractive summarization.
- **Embeddings:** Integrated BioWordVec to capture domain-specific medical semantics.
- **Optimization:** Tuned hyperparameters with GridSearchCV; implemented beam search decoding.
- **Evaluation:** Improved ROUGE-1/ROUGE-L scores compared to baseline seq2seq models.

⸻

## Results
	•	Produced concise summaries for complex CHQs with reduced noise.
	•	Attention visualization improved interpretability of generated outputs.
	•	Identified limitations such as hallucination & repetition, with recommendations for future transformer-based GenAI approaches.

⸻

## Impact & Use Cases
	•	Enables medical researchers to analyze patient queries faster.
	•	Enhances healthcare chatbots with GenAI-powered summarization.
	•	Reduces information overload in patient-facing platforms.

⸻

## Future Work
	•	Extend to transformer-based GenAI models (BERT, T5, Pegasus) for improved results.
	•	Explore reinforcement learning approaches to mitigate hallucinations.
	•	Scale to broader healthcare datasets for generalizability.

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

