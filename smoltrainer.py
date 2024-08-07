import torch
import torch.nn as nn
import logging
import json
from datasets import load_dataset, Dataset, concatenate_datasets
from transformers import AutoConfig, AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer, DataCollatorForLanguageModeling
from torch.cuda.amp import autocast
import warnings
from tqdm import tqdm
import math
from huggingface_hub import HfFolder, Repository, upload_folder

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

MODEL_NAME = "DewEfresh/SmolLM-135M-merged-v3"
MAX_LENGTH = 2048
BATCH_SIZE = 2
LEARNING_RATE = 2e-4
MAX_STEPS = 3000
GRADIENT_ACCUMULATION_STEPS = 2
NUM_WARMUP_STEPS = 30
OUTPUT_DIR = "./longcustom_finetuned_results"
CUSTOM_DATASET_PATH = "dummydataset.jsonl"
HF_MODEL_NAME = "your-username/your-model-name"  # Replace with your desired Hugging Face model name

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

# ... [rest of the code remains the same] ...

def upload_to_hf(local_dir, repo_name):
    logger.info(f"🚀 Uploading model to Hugging Face Hub: {repo_name}")
    try:
        token = HfFolder.get_token()
        if token is None:
            logger.error("❌ Hugging Face token not found. Please log in using `huggingface-cli login`")
            return False

        repo = Repository(local_dir, clone_from=repo_name, use_auth_token=token)
        repo.git_pull()

        # Add all files in the directory
        repo.git_add()

        # Commit and push
        repo.git_commit("Update model")
        repo.git_push()

        logger.info(f"✅ Successfully uploaded model to {repo_name}")
        return True
    except Exception as e:
        logger.error(f"❌ Failed to upload model to Hugging Face: {str(e)}")
        return False

def main():
    logger.info(f"🚀 Initializing {MODEL_NAME} finetuning with GrokAdamW")
    
    # ... [rest of the main function remains the same] ...

    logger.info("💾 Saving the model")
    try:
        trainer.save_model(OUTPUT_DIR)
        tokenizer.save_pretrained(OUTPUT_DIR)
    except Exception as e:
        logger.error(f"❌ Failed to save model: {str(e)}")
        return

    logger.info("🎉 Finetuning with GrokAdamW completed!")

    # Upload to Hugging Face
    if upload_to_hf(OUTPUT_DIR, HF_MODEL_NAME):
        logger.info(f"🌟 Model successfully uploaded to Hugging Face: {HF_MODEL_NAME}")
    else:
        logger.warning("⚠️ Failed to upload model to Hugging Face")

if __name__ == "__main__":
    main()
