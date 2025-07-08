🔗 👉 **[Watch the Demo on YouTube](https://www.youtube.com/watch?v=TTt2aFM7G5c&list=PLe-YIIlt-fbPMDsmSXbzQuyBeRKfvs__T&index=2&ab_channel=Jatin)**
-----

# 📝 Abstractive Text Summarization

**Fine-tuning Hugging Face Models for High-Quality Text Summarization**

This project provides an end-to-end MLOps pipeline for building and deploying an abstractive text summarization model. It leverages the power of Hugging Face Transformers to fine-tune a pre-trained model (likely Pegasus) on a conversational dataset, enabling it to generate concise summaries from longer texts. The pipeline covers data ingestion, transformation, model training, evaluation, and includes a prediction interface.

-----

## 🎯 Project Overview

In today's information-rich world, automatic text summarization is invaluable for quickly grasping the essence of large documents or conversations. This project focuses on **abstractive summarization**, where the model generates new sentences that capture the core meaning, rather than just extracting existing ones. By fine-tuning a state-of-the-art Hugging Face model, we achieve high-quality, context-aware summaries.

**Key Objectives:**

  * **Build a High-Performance Summarizer:** Fine-tune a pre-trained Transformer model to generate accurate and coherent summaries.
  * **Establish an MLOps Pipeline:** Implement a structured workflow for data handling, model training, and evaluation.
  * **Leverage Hugging Face Ecosystem:** Utilize `transformers` and `datasets` libraries for efficient model and data management.
  * **Ensure Reproducibility:** Organize the project with configuration management for consistent runs.

-----

## ✨ Key Features

  * **Abstractive Summarization:** Generates new, concise summaries rather than merely extracting sentences.
  * **Hugging Face Transformers:** Utilizes powerful pre-trained sequence-to-sequence models (e.g., Pegasus) from Hugging Face.
  * **Fine-tuning Pipeline:** Includes structured stages for:
      * **Data Ingestion:** Automatically downloads and extracts the dataset (e.g., Samsum).
      * **Data Transformation:** Tokenizes and prepares the data for model training, aligning with the Transformer's requirements.
      * **Model Training:** Fine-tunes the pre-trained model on the specific summarization task.
      * **Model Evaluation:** Assesses the model's performance using standard metrics like ROUGE scores.
  * **Prediction Pipeline:** Provides a robust way to generate summaries for new input texts.
  * **Configuration Management:** Uses `config.yaml` and `params.yaml` to manage settings and hyperparameters, enhancing reproducibility.
  * **Modular Codebase:** Organized into a clean `src` directory structure for maintainability and scalability.
  * **Trained on Conversational Data:** Optimized for summarizing dialogues, making it highly relevant for customer service logs, meeting transcripts, etc.

-----

## ⚙️ Pipeline Stages & Technologies

This project is structured as an MLOps pipeline, with distinct components for each stage:

### **1. Data Ingestion**

  * **Purpose:** Downloads the raw dataset (e.g., a `.zip` file containing dialogue-summary pairs) from a specified URL and extracts it.
  * **Key Operations:**
      * Downloads data if not already present.
      * Extracts contents of the zipped file.
  * **Technologies:** `urllib.request`, `zipfile`, `os`.

### **2. Data Transformation**

  * **Purpose:** Prepares the raw text data for the Transformer model. This involves tokenization and formatting.
  * **Key Operations:**
      * Loads the dataset from disk.
      * Initializes a tokenizer from the specified pre-trained model.
      * Converts dialogue and summary examples into numerical input IDs and attention masks, suitable for the Transformer model.
      * Pads and truncates sequences to fixed lengths (e.g., 1024 for input, 128 for target summaries).
      * Saves the processed dataset.
  * **Technologies:** `Hugging Face AutoTokenizer`, `Hugging Face datasets`.

### **3. Model Training**

  * **Purpose:** Fine-tunes a pre-trained sequence-to-sequence model on the processed summarization dataset.
  * **Key Operations:**
      * Loads the pre-trained model and tokenizer from Hugging Face.
      * Uses `DataCollatorForSeq2Seq` to prepare batches for training.
      * Sets up `TrainingArguments` (e.g., number of epochs, batch size, logging steps).
      * Initializes and runs the `Trainer` for fine-tuning.
      * Saves the fine-tuned model and tokenizer.
  * **Technologies:** `Hugging Face AutoModelForSeq2SeqLM`, `Hugging Face Trainer`, `TrainingArguments`, `DataCollatorForSeq2Seq`, `PyTorch`.

### **4. Model Evaluation**

  * **Purpose:** Assesses the performance of the fine-tuned model on a test dataset.
  * **Key Operations:**
      * Loads the fine-tuned model and tokenizer.
      * Generates summaries for a batch of test dialogues.
      * Calculates ROUGE scores (ROUGE-1, ROUGE-2, ROUGE-L, ROUGE-Lsum), which measure the overlap between generated and reference summaries.
      * Saves the evaluation metrics to a CSV file.
  * **Technologies:** `Hugging Face evaluate`, `pandas`, `tqdm`.

### **5. Prediction Pipeline**

  * **Purpose:** Provides an interface to use the fine-tuned model for generating summaries from new, unseen text.
  * **Key Operations:**
      * Loads the tokenizer and model.
      * Uses Hugging Face's `pipeline` utility for streamlined summarization.
      * Applies generation parameters (e.g., `length_penalty`, `num_beams`, `max_length`) for better summary quality.
  * **Technologies:** `Hugging Face pipeline`.

### **Configuration Management**

  * **Purpose:** Centralizes all configurations and parameters (e.g., data paths, model names, training arguments) in `config.yaml` and `params.yaml` files.
  * **Benefit:** Makes the project easily configurable, reproducible, and scalable.
  * **Technologies:** Custom `ConfigurationManager` class, `PyYAML`.

-----

## 📂 Project Structure

```
.
├── artifacts/                      # Stores pipeline outputs (data, models, metrics)
│   ├── data_ingestion/             # Downloaded and unzipped raw data
│   ├── data_transformation/        # Processed (tokenized, padded) dataset
│   ├── model_trainer/              # Fine-tuned model and tokenizer
│   ├── model_evaluation/           # Evaluation metrics (ROUGE scores)
│   └── ...
├── src/                            # Source code for the MLOps pipeline stages
│   ├── textSummarizer/
│   │   ├── components/             # Individual pipeline components
│   │   │   ├── data_ingestion.py
│   │   │   ├── data_transformation.py
│   │   │   ├── model_evaluation.py
│   │   │   ├── model_trainer.py
│   │   │   └── __init__.py
│   │   ├── config/                 # Configuration management
│   │   │   ├── configuration.py
│   │   │   └── __init__.py
│   │   ├── entity/                 # Data class definitions
│   │   ├── pipeline/               # Orchestration logic
│   │   │   ├── prediction.py
│   │   │   ├── stage_01_data_ingestion.py
│   │   │   ├── stage_02_data_transformation.py
│   │   │   ├── stage_03_model_trainer.py
│   │   │   ├── stage_04_model_evaluation.py
│   │   │   └── __init__.py
│   │   ├── utils/                  # Helper functions
│   │   │   └── common.py
│   │   ├── logging/                # Custom logging
│   │   └── __init__.py
│   ├── main.py                     # Pipeline runner
│   ├── app.py                      # Optional Streamlit/Flask UI
│   └── requirements.txt            # Dependencies
├── config/                         # Configuration files
│   ├── config.yaml
│   └── params.yaml
├── research/                       # Jupyter notebooks
│   └── notebooks.ipynb
├── README.md                       # This file
└── setup.py                        # Packaging info
```

-----

## 🚀 How to Run Locally

To set up and run this project on your local machine, follow these steps:

1.  **Clone the Repository:**

    ```bash
    git clone <your-repository-url>
    cd <your-project-folder>
    ```

2.  **Create a Virtual Environment (Recommended):**

    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows: `venv\Scripts\activate`
    ```

3.  **Install Dependencies:**

    ```bash
    pip install -r requirements.txt
    ```

4.  **Configure Project:**

      * Review and modify `config/config.yaml` to specify data paths, model checkpoints, etc.
      * Review and modify `config/params.yaml` to adjust model training hyperparameters (e.g., `num_train_epochs`, `per_device_train_batch_size`).

5.  **Run the MLOps Pipeline:**
    Execute the main pipeline script. This will perform data ingestion, transformation, model training, and evaluation sequentially.

    ```bash
    python main.py
    ```

6.  **Make Predictions:**
    You can use the prediction pipeline by importing it, or integrate it into a `main.py` if desired.

    ```python
    # Example usage (can be added to a script or run in a Python console)
    from src.textSummarizer.pipeline.prediction import PredictionPipeline

    predictor = PredictionPipeline()
    text_to_summarize = "Dialogue: This is a long conversation between two people. They talk about their plans for the weekend, including going to the park and having a picnic. Speaker A suggests bringing sandwiches, and Speaker B agrees, also mentioning drinks and snacks. They confirm the time and place. Summary: Speaker A and B are planning a weekend picnic in the park, agreeing on food and logistics." # Replace with your actual dialogue
    summary = predictor.predict(text_to_summarize)
    print(f"Generated Summary: {summary}")
    ```

-----

## 📝 Usage and Customization

  * **Dataset:** Currently configured for a specific dataset (likely Samsum, based on `dataset_samsum`), but can be adapted to other summarization datasets by updating `config.yaml` and `DataIngestion` logic.
  * **Base Model:** The project uses a model from Hugging Face (e.g., `google/pegasus-cnn_dailymail` or similar). You can change the `model_ckpt` in `config.yaml` to fine-tune other pre-trained sequence-to-sequence models (e.g., T5, BART).
  * **Hyperparameters:** Adjust training parameters in `params.yaml` to optimize model performance for your specific needs.
  * **Scalability:** The modular structure makes it easier to integrate with cloud-based MLOps platforms (e.g., MLflow, Kubeflow) for large-scale training and deployment.

-----

## 📈 Evaluation Metrics

The model's performance is evaluated using **ROUGE (Recall-Oriented Understudy for Gisting Evaluation)** scores. ROUGE is a set of metrics used for evaluating automatic summarization and machine translation. It works by comparing an automatically produced summary or translation against a set of human-produced reference summaries or translations.

  * **ROUGE-1:** Measures the overlap of unigrams (single words) between the generated and reference summaries.
  * **ROUGE-2:** Measures the overlap of bigrams (pairs of words) between the generated and reference summaries.
  * **ROUGE-L:** Measures the longest common subsequence (LCS) match between the summaries, capturing sentence-level structure.
  * **ROUGE-Lsum:** Similar to ROUGE-L, but calculated over entire summaries (multi-sentence).

The `ModelEvaluation` component calculates these scores and saves them to a CSV file.

-----

## 🤝 Credits

  * [Jatin Yadav]
  * [Hugging Face Transformers](https://huggingface.co/docs/transformers/index)
  * [Hugging Face Datasets](https://huggingface.co/docs/datasets/index)
  * [PyTorch](https://pytorch.org/)
  * [Samsum Dataset](https://huggingface.co/datasets/samsum) (Assumed dataset /  I took from someone else's profile)

-----

## 🙋‍♂️ Let's Connect

* **💼 LinkedIn:** [www.linkedin.com/in/jatin557](https://www.linkedin.com/in/jatin557)
* **📦 GitHub:** [https://github.com/jatinydav557](https://github.com/jatinydav557)
* **📬 Email:** [jatinydav557@gmail.com](mailto:jatinydav557@gmail.com)
* **📱 Contact:** [`+91-7340386035`](tel:+917340386035)
* **🎥 YouTube:** [Checkout my other working projects](https://www.youtube.com/@jatinML/playlists)

It was a complex project.I hope you watch the demo video 
