import torch
from transformers import AutoTokenizer, BartForConditionalGeneration, BartConfig
import src.utils as log
import os
from src.check_gpu import get_device

def load_tokenizer_and_model_for_train(cfg):
    log.info('-'*10 + ' Load tokenizer & model (Train) ' + '-'*10)
    device = get_device()
    model_name = cfg.general.model_name
    log.info(f'Model Name : {model_name}')
    bart_config = BartConfig().from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    generate_model = BartForConditionalGeneration.from_pretrained(model_name, config=bart_config)

    special_tokens_dict={'additional_special_tokens':cfg.tokenizer.special_tokens}
    tokenizer.add_special_tokens(special_tokens_dict)

    generate_model.resize_token_embeddings(len(tokenizer))
    generate_model.to(device)
    log.info(generate_model.config)

    log.info('-'*10 + ' Load tokenizer & model (Train) complete ' + '-'*10)
    return generate_model , tokenizer

def load_tokenizer_and_model_for_test(cfg):
    log.info('-'*10 + ' Load tokenizer & model (Test) ' + '-'*10)
    device = get_device()

    model_name = cfg.general.model_name
    ckt_path = cfg.inference.ckt_path
    log.info(f'Model Name : {model_name}')
    log.info(f'Checkpoint Path : {ckt_path}')

    tokenizer = AutoTokenizer.from_pretrained(model_name) # local_files_only=True 제거
    special_tokens_dict = {'additional_special_tokens': cfg.tokenizer.special_tokens}
    tokenizer.add_special_tokens(special_tokens_dict)

    generate_model = BartForConditionalGeneration.from_pretrained(os.path.abspath(ckt_path), local_files_only=True)
    generate_model.resize_token_embeddings(len(tokenizer))
    generate_model.to(device)
    log.info('-'*10 + ' Load tokenizer & model (Test) complete ' + '-'*10)

    return generate_model , tokenizer
