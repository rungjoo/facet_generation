import os
import pdb
import argparse, logging
from tqdm import tqdm

import torch
from transformers import get_linear_schedule_with_warmup

from dataloader import data_loader
from torch.utils.data import DataLoader

from transformers import AutoTokenizer, BartForConditionalGeneration
    
## finetune gpt2
def main():    
    device = torch.device("cuda" if torch.cuda.is_available() else 'cpu')
    
    """Dataset Loading"""    
    batch_size = args.batch
    tokenizer_path = "/home/jovyan/hdfs-jmt-rungjoo-private/huggingface_models/bart-base"
    
    data_type = args.data_type
    train_path = "../data/merge_train.json"
    train_dataset = data_loader(train_path, tokenizer_path, data_type)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, collate_fn=train_dataset.collate_fn)
    
    """logging and path"""    
    save_path = f"/home/jovyan/hdfs-jmt-rungjoo-private/save_models/facet/query_pick_document_bart"
    print("###Save Path### ", save_path)
    log_path = "train.log"
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    fileHandler = logging.FileHandler(log_path)
    
    logger.addHandler(streamHandler)
    logger.addHandler(fileHandler)    
    logger.setLevel(level=logging.DEBUG)      
    
    logger.info('Batch size: {}'.format(batch_size))
    
    """Model Loading"""
    model_path = "/home/jovyan/hdfs-jmt-rungjoo-private/huggingface_models/bart-base"
    model = BartForConditionalGeneration.from_pretrained(model_path)
    model = model.cuda()
    model.train()
    
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
    train_dataset.tokenizer.save_pretrained(save_path)
    
    for epoch in tqdm(range(training_epochs)):
        model.train() 
        for i_batch, inputs in enumerate(tqdm(train_dataloader)):            
            """Prediction"""
            enc_input_ids, enc_attention_mask, dec_input_ids, dec_attention_mask, labels = inputs
            
            enc_input_ids = enc_input_ids.to(device)
            enc_attention_mask = enc_attention_mask.to(device)
            dec_input_ids = dec_input_ids.to(device)
            dec_attention_mask = dec_attention_mask.to(device)
            labels = labels.to(device)
            
            outputs = model(
                input_ids=enc_input_ids,
                attention_mask=enc_attention_mask,
                decoder_input_ids=dec_input_ids,
                decoder_attention_mask=dec_attention_mask,
                labels=labels
            )            
            
            """Loss calculation & training"""
            loss_val = outputs.loss
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
    # torch.save(model.state_dict(), os.path.join(path, 'pytorch_model.bin'))
    model.save_pretrained(path)
    

if __name__ == '__main__':
    torch.cuda.empty_cache()
    
    """Parameters"""
    parser  = argparse.ArgumentParser(description = "block ranking system" )
    parser.add_argument( "--batch", type=int, help = "batch_size", default = 1)
    parser.add_argument( "--epoch", type=int, help = 'training epohcs', default = 10)
    parser.add_argument( "--norm", type=int, help = "max_grad_norm", default = 10)
    parser.add_argument( "--lr", type=float, help = "learning rate", default = 1e-6) # 1e-5
    parser.add_argument( "--data_type", type=str, help = "data_type", default = 'original')
        
    args = parser.parse_args()
    
    logger = logging.getLogger(__name__)
    streamHandler = logging.StreamHandler()
    
    main()