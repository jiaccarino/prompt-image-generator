import qai_hub
import subprocess
import logging
import argparse

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('model_id_fetcher.log'),
        logging.StreamHandler()
    ]
)

def fetch_model_ids(batch_size=5, max_models=None):
    """Fetch model IDs from QAI Hub."""
    logging.info(f"Fetching model IDs with batch size: {batch_size}")
    model_ids = []
    offset = 0
    try:
        while True:
            batch = qai_hub.get_models(limit=batch_size, offset=offset)
            if not batch:
                break
            model_ids.extend([model.model_id for model in batch])
            offset += batch_size
            if len(batch) < batch_size or (max_models and len(model_ids) >= max_models):
                break
        if max_models:
            model_ids = model_ids[:max_models]
        logging.info(f"Fetched {len(model_ids)} model IDs")
        return model_ids
    except Exception as e:
        logging.error(f"Error fetching model IDs: {str(e)}")
        raise

def run_main_script(model_ids, device=None):
    """Run the main script with fetched model IDs."""
    cmd = ["python", "qaihub_batch_model_inputs.py", "--model_ids"] + model_ids
    if device:
        cmd.extend(["--device", device])
    
    logging.info(f"Running main script with command: {' '.join(cmd)}")
    try:
        subprocess.run(cmd, check=True)
        logging.info("Main script completed successfully")
    except subprocess.CalledProcessError as e:
        logging.error(f"Error running main script: {str(e)}")
        raise

def main():
    parser = argparse.ArgumentParser(description="Fetch model IDs and run the main processing script")
    parser.add_argument("--max_models", type=int, help="Maximum number of models to process")
    parser.add_argument("--device", help="Specific device to use for all models")
    args = parser.parse_args()

    try:
        model_ids = fetch_model_ids(max_models=args.max_models)
        run_main_script(model_ids, args.device)
    except Exception as e:
        logging.error(f"Script execution failed: {str(e)}")

if __name__ == "__main__":
    main()
