import os
import torch
import pandas as pd
import src.utils as log
from tqdm import tqdm
from src.model import load_tokenizer_and_model_for_test
from src.data import Preprocess, prepare_test_dataset
from src.check_gpu import get_device 

def inference(cfg):
    device = get_device()
    log.info(f"PyTorch version: {torch.__version__}") 
    generate_model , tokenizer = load_tokenizer_and_model_for_test(cfg) 

    preprocessor = Preprocess(cfg.tokenizer.bos_token, cfg.tokenizer.eos_token)
    test_data, test_encoder_inputs_dataset = prepare_test_dataset(cfg, preprocessor, tokenizer)
    dataloader = torch.utils.data.DataLoader(test_encoder_inputs_dataset, batch_size=cfg.inference.batch_size)

    summary = []
    text_ids = []
    with torch.no_grad():
        for item in tqdm(dataloader):
            text_ids.extend(item['ID'])
            generated_ids = generate_model.generate(input_ids=item['input_ids'].to(device),
                            no_repeat_ngram_size=cfg.inference.no_repeat_ngram_size,
                            early_stopping=cfg.inference.early_stopping,
                            max_length=cfg.inference.generate_max_length,
                            num_beams=cfg.inference.num_beams,
                        )
            for ids in generated_ids:
                result = tokenizer.decode(ids)
                summary.append(result)

    remove_tokens = cfg.inference.remove_tokens
    preprocessed_summary = summary.copy()
    for token in remove_tokens:
        preprocessed_summary = [sentence.replace(token," ") for sentence in preprocessed_summary]

    output = pd.DataFrame({
        "fname": test_data['fname'],
        "summary" : preprocessed_summary,
        })
    result_path = cfg.inference.result_path
    if not os.path.exists(result_path):
        os.makedirs(result_path)
    output.to_csv(os.path.join(result_path, "output.csv"), index=False)

    return output
