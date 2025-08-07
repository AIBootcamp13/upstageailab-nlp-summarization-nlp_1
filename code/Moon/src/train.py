import os
import torch
from rouge import Rouge
from transformers import Seq2SeqTrainingArguments, Seq2SeqTrainer, EarlyStoppingCallback, TrainerCallback
import wandb
from dotenv import load_dotenv
import src.utils as log

# .env 파일 로드
load_dotenv()

class LoggingCallback(TrainerCallback):
    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs and 'eval_loss' in logs:
            log.info(f"Evaluation metrics: {logs}")

def compute_metrics(cfg, tokenizer, pred):
    rouge = Rouge()
    predictions = pred.predictions
    labels = pred.label_ids

    predictions[predictions == -100] = tokenizer.pad_token_id
    labels[labels == -100] = tokenizer.pad_token_id

    decoded_preds = tokenizer.batch_decode(predictions, clean_up_tokenization_spaces=True)
    labels = tokenizer.batch_decode(labels, clean_up_tokenization_spaces=True)

    remove_tokens = cfg.inference.remove_tokens
    replaced_predictions = decoded_preds.copy()
    replaced_labels = labels.copy()
    for token in remove_tokens:
        replaced_predictions = [sentence.replace(token," ") for sentence in replaced_predictions]
        replaced_labels = [sentence.replace(token," ") for sentence in replaced_labels]

    log.info('-'*150)
    log.info(f"PRED: {replaced_predictions[0]}")
    log.info(f"GOLD: {replaced_labels[0]}")
    log.info('-'*150)
    log.info(f"PRED: {replaced_predictions[1]}")
    log.info(f"GOLD: {replaced_labels[1]}")
    log.info('-'*150)
    log.info(f"PRED: {replaced_predictions[2]}")
    log.info(f"GOLD: {replaced_labels[2]}")
    log.info('-'*150)

    results = rouge.get_scores(replaced_predictions, replaced_labels, avg=True)
    
    log.info("ROUGE Evaluation Results:")
    for metric_name, metric_values in results.items():
        log.info(f"{metric_name.upper()}: Precision={metric_values['p']:.4f}, Recall={metric_values['r']:.4f}, F1={metric_values['f']:.4f}")
    log.info('-'*150)

    result = {key: value["f"] for key, value in results.items()}
    return result

def load_trainer_for_train(cfg, generate_model, tokenizer, train_inputs_dataset, val_inputs_dataset):
    log.info('-'*10 + ' Make training arguments ' + '-'*10)
    
    training_args = Seq2SeqTrainingArguments(
                output_dir=cfg.general.output_dir,
                overwrite_output_dir=cfg.training.overwrite_output_dir,
                num_train_epochs=cfg.training.num_train_epochs,
                learning_rate=cfg.training.learning_rate,
                per_device_train_batch_size=cfg.training.per_device_train_batch_size,
                per_device_eval_batch_size=cfg.training.per_device_eval_batch_size,
                warmup_ratio=cfg.training.warmup_ratio,
                weight_decay=cfg.training.weight_decay,
                lr_scheduler_type=cfg.training.lr_scheduler_type,
                optim=cfg.training.optim,
                gradient_accumulation_steps=cfg.training.gradient_accumulation_steps,
                eval_strategy=cfg.training.eval_strategy,
                save_strategy=cfg.training.save_strategy,
                save_total_limit=cfg.training.save_total_limit,
                fp16=cfg.training.fp16,
                load_best_model_at_end=cfg.training.load_best_model_at_end,
                seed=cfg.training.seed,
                logging_dir=cfg.training.logging_dir,
                logging_strategy=cfg.training.logging_strategy,
                predict_with_generate=cfg.training.predict_with_generate,
                generation_max_length=cfg.training.generation_max_length,
                do_train=cfg.training.do_train,
                do_eval=cfg.training.do_eval,
                report_to=cfg.training.report_to
            )

    use_wandb = os.getenv('USE_WANDB', '').lower() == 'true'
    if cfg.training.report_to == 'wandb' and use_wandb:
        os.environ["WANDB_DIR"] = "../"

        wandb.init(
            entity=cfg.training.wandb.entity,
            project=cfg.training.wandb.project,
            name=cfg.training.wandb.name,
        )
        os.environ["WANDB_LOG_MODEL"]="true"
        os.environ["WANDB_WATCH"]="false"
    else:
        # wandb를 사용하지 않으므로 report_to를 None으로 변경
        training_args.report_to = None
        os.environ['WANDB_DISABLED'] = 'true' # wandb를 완전히 비활성화

    MyCallback = EarlyStoppingCallback(
        early_stopping_patience=cfg.training.early_stopping_patience,
        early_stopping_threshold=cfg.training.early_stopping_threshold
    )
    log.info('-'*10 + ' Make training arguments complete ' + '-'*10)
    log.info('-'*10 + ' Make trainer ' + '-'*10)

    trainer = Seq2SeqTrainer(
        model=generate_model,
        args=training_args,
        train_dataset=train_inputs_dataset,
        eval_dataset=val_inputs_dataset,
        compute_metrics = lambda pred: compute_metrics(cfg, tokenizer, pred),
        callbacks = [MyCallback, LoggingCallback()]
    )
    log.info('-'*10 + ' Make trainer complete ' + '-'*10)

    return trainer
