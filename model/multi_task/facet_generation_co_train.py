import os, sys
import pdb
import argparse, logging
from tqdm import tqdm

import torch
from transformers import get_linear_schedule_with_warmup

from dataloader_co import data_loader
from torch.utils.data import DataLoader

from transformers import AutoTokenizer, BartForConditionalGeneration
    
## finetune gpt2
def main():    
    device = torch.device("cuda" if torch.cuda.is_available() else 'cpu')
    if args.document:
        document = "document"
    else:
        document = ""
    if args.related:
        related = "related"
    else:
        related = ""
    rationale = "rationale"
    task_name = f"co_{document}_{related}_{rationale}".strip("_")    
    # task_name = "rationale_baseline"
    if task_name == "":
        print("멀티테스크를 입력하세요")
        sys.exit()
    
    """Dataset Loading"""    
    batch_size = args.batch
    tokenizer_path = "/home/jovyan/hdfs-jmt-rungjoo-private/huggingface_models/bart-base"
    
    train_path = "../../data/train_rationale_7B.json"
    train_dataset = data_loader(train_path, tokenizer_path)    
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False, num_workers=4, collate_fn=train_dataset.collate_fn)
    
    """logging and path"""    
    save_path = f"/home/jovyan/hdfs-jmt-rungjoo-private/save_models/facet/multi_task/{task_name}"
    print("###Save Path### ", save_path)
    log_path = "train.log"
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    fileHandler = logging.FileHandler(log_path)
    
    logger.addHandler(streamHandler)
    logger.addHandler(fileHandler)    
    logger.setLevel(level=logging.DEBUG)      
    
    logger.info('Task name: {}'.format(task_name))
    logger.info('Batch size: {}'.format(batch_size))
    
    """Model Loading"""
    model_path = "/home/jovyan/hdfs-jmt-rungjoo-private/huggingface_models/bart-base"
    model = BartForConditionalGeneration.from_pretrained(model_path)
    model = model.cuda()
    model.train()
    model.resize_token_embeddings(len(train_dataset.tokenizer))
    
    """Training Setting"""        
    training_epochs = args.epoch
    save_term = int(training_epochs/5)
    max_grad_norm = args.norm
    lr = args.lr
    num_training_steps = len(train_dataset)*training_epochs
    num_warmup_steps = len(train_dataset)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr) # , eps=1e-06, weight_decay=0.01
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=num_warmup_steps, num_training_steps=num_training_steps)
        
    """Input & Label Setting"""
    def allocate_device(model_input):
        model_device_input = {}
        for k, v in model_input.items():
            model_device_input[k] = v.to(device)
        return model_device_input
    
    train_dataset.tokenizer.save_pretrained(save_path)
    
    for epoch in tqdm(range(training_epochs)):
        model.train() 
        for i_batch, inputs in enumerate(tqdm(train_dataloader)):            
            """Multi-task prediction"""
            f_input, d_input, r_input = inputs            
            f_input = allocate_device(f_input)
            d_input = allocate_device(d_input)
            r_input = allocate_device(r_input)
            
            f_outputs = model(
                input_ids=f_input["enc_input_ids"],
                attention_mask=f_input["enc_attention_mask"],
                decoder_input_ids=f_input["dec_input_ids"],
                decoder_attention_mask=f_input["dec_attention_mask"],
                labels=f_input["labels"]
            )
            
            if args.document:
                d_outputs = model(
                    input_ids=d_input["enc_input_ids"],
                    attention_mask=d_input["enc_attention_mask"],
                    decoder_input_ids=d_input["dec_input_ids"],
                    decoder_attention_mask=d_input["dec_attention_mask"],
                    labels=d_input["labels"]
                )
            
            if args.related:
                r_outputs = model(
                    input_ids=r_input["enc_input_ids"],
                    attention_mask=r_input["enc_attention_mask"],
                    decoder_input_ids=r_input["dec_input_ids"],
                    decoder_attention_mask=r_input["dec_attention_mask"],
                    labels=r_input["labels"]
                )
            
            """Loss calculation & training"""
            loss_val = f_outputs.loss
            if args.document:
                loss_val += d_outputs.loss
            if args.related:
                loss_val += r_outputs.loss
            loss_val.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)  # Gradient clipping is not in AdamW anymore (so you can use amp without issue)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
        
        """Best Score & Model Save"""
        logger.info('Epoch: {}'.format(epoch))
        _SaveModel(model, save_path)    
        
def _SaveModel(model, path):
    if not os.path.exists(path):
        os.makedirs(path)
    model.save_pretrained(path)
    

if __name__ == '__main__':
    torch.cuda.empty_cache()
    
    """Parameters"""
    parser  = argparse.ArgumentParser(description = "block ranking system" )
    parser.add_argument( "--batch", type=int, help = "batch_size", default = 1)
    parser.add_argument( "--epoch", type=int, help = 'training epohcs', default = 10)
    parser.add_argument( "--norm", type=int, help = "max_grad_norm", default = 10)
    parser.add_argument( "--lr", type=float, help = "learning rate", default = 1e-6) # 1e-5
    
    parser.add_argument('--document', action='store_true', help='train document')
    parser.add_argument('--related', action='store_true', help='train related')
    parser.add_argument('--rationale', action='store_true', help='train rationale')
        
    args = parser.parse_args()
    
    logger = logging.getLogger(__name__)
    streamHandler = logging.StreamHandler()
    
    main()