# env_utils.py
import random
import logging
import numpy as np
import torch
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo
from contextlib import contextmanager
import os
import ray
from verl.utils.fs import copy_to_local

def permanent_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


@contextmanager
def set_seed(seed):
    random_state = random.getstate()
    np_random_state = np.random.get_state()

    try:
        random.seed(seed)
        np.random.seed(seed)
        yield
    finally:
        random.setstate(random_state)
        np.random.set_state(np_random_state)


def setup_logging(output_dir):
    os.makedirs(output_dir, exist_ok=True)
    log_file = os.path.join(output_dir, 'training.log')
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
        handlers=[logging.FileHandler(log_file), logging.StreamHandler()]
    )
    logging.Formatter.converter = lambda *args: (datetime.now(UTC) - timedelta(hours=2)).timetuple()
    return logging.getLogger()


@contextmanager
def NoLoggerWarnings():
    from gymnasium import logger
    logger.set_level(logger.ERROR)
    try:
        yield
    finally:
        logger.set_level(logger.INFO)


def get_train_val_env(env_class, config: dict):
    if 'medical_consultation' in config.env.name:
        # Initialize environment LLM worker if needed
        env_llm_worker = None
        tokenizer = None
        
        if config.env.get('use_env_llm', False):
            from ragen.workers.env_llm_worker import EnvironmentLLMWorker
            from transformers import AutoTokenizer
            import ray
            
            # Get the env_llm_worker from Ray
            # env_llm_worker = ray.get_actor('env_llm')
            
            # Initialize tokenizer for the environment LLM
            local_path = copy_to_local(config.env.env_llm.model.path)
            tokenizer = AutoTokenizer.from_pretrained(
                local_path,
                trust_remote_code=config.env.env_llm.model.get('trust_remote_code', False)
            )
            
            print(f"[INFO] Using Ray-managed environment LLM worker")
        
        env = env_class(
            parquet_path=config.data.train_files,
            env_llm_worker=env_llm_worker,
            tokenizer=tokenizer,
            max_turns=config.env.get('max_turns', 5)
        )
        
        val_env = env_class(
            parquet_path=config.data.val_files,
            env_llm_worker=env_llm_worker,
            tokenizer=tokenizer,
            max_turns=config.env.get('max_turns', 5)
        )
    else:
        raise ValueError(f"Only medical consultation environments are supported, got: {config.env.name}")

    return env, val_env
