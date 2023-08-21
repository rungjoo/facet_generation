from torch.utils.data import Dataset
from transformers import AutoTokenizer
import torch, json
    
class data_loader(Dataset):
    def __init__(self, data_path, tokenizer_path):
        with open(data_path, 'r', encoding='utf-8') as f:
            self.dataset = json.load(f)
        
        tokenizer_path = tokenizer_path
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
        if self.tokenizer.pad_token_id == None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        # self.tokenizer.padding_side = "left"
        
    def __len__(self): # 기본적인 구성
        return len(self.dataset)
    
    def __getitem__(self, idx): # 기본적인 구성
        return self.dataset[str(idx)]
    
    def collate_fn(self, batch_data): # 배치를 위한 구성
        '''
            input:
                batch_data: [(query, title), (query, title), ... ]
            return:
                batch_padding_input_token: (B, L) padded
                batch_padding_attention_mask:
        '''
        batch_enc = []
        batch_dec = []
        for data in batch_data:
            input_string = data['query']
            batch_enc.append(input_string)
                        
            facet = data['facet']
            taget_string = ", ".join(facet).strip()
            batch_dec.append(taget_string)
            
        enc_inputs = self.tokenizer(batch_enc, padding=True, truncation=True, max_length=self.tokenizer.model_max_length, return_tensors='pt', add_special_tokens=False)
        dec_inputs = self.tokenizer(batch_dec, padding=True, truncation=True, max_length=self.tokenizer.model_max_length, return_tensors='pt')

        enc_input_ids = enc_inputs['input_ids']
        enc_attention_mask = enc_inputs['attention_mask']

        dec_input_ids = dec_inputs['input_ids']        
        real_dec_input_ids = dec_input_ids[:, :-1]
        dec_attention_mask = dec_inputs['attention_mask'][:,:-1]
        
        labels = dec_input_ids[:, 1:].clone()

        return enc_input_ids, enc_attention_mask, real_dec_input_ids, dec_attention_mask, labels