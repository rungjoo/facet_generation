from torch.utils.data import Dataset
from transformers import AutoTokenizer
import torch, json
    
class data_loader(Dataset):
    def __init__(self, data_path, tokenizer_path, pick_num):
        with open(data_path, 'r', encoding='utf-8') as f:
            original_dataset = json.load(f)
            
        self.dataset = {}
        num = 0
        for k, data in original_dataset.items():
            # document와 related가 있는 경우 (SERP 데이터에 query가 있는 경우임)
            if ('pick_document' in data) and ('pick_related' in data):
                facet = data['facet']
                facet_num = len(facet)
                if pick_num == "auto":
                    document = [x[0] for x in data['pick_document'][:facet_num]]
                    related = [x[0] for x in data['pick_related'][:facet_num]]
                else:
                    pick_num = int(pick_num)
                    document = [x[0] for x in data['pick_document'][:pick_num]]
                    related = [x[0] for x in data['pick_related'][:pick_num]]
                
                # documet와 related가 한 개 이상 할당된 경우
                # SERP라고 해서 모두 document, related가 태깅되어 있진 않음
                if len(document) > 0 or len(related) > 0:
                    self.dataset[num] = {}
                    self.dataset[num]['query'] = data['query']
                    self.dataset[num]['facet'] = facet
                    self.dataset[num]['document'] = document
                    self.dataset[num]['related'] = related
                    num += 1
        
        tokenizer_path = tokenizer_path
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
        if self.tokenizer.pad_token_id == None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            
        special_tokens_dict = {'additional_special_tokens': ['[facet]', '[document]', '[related]']}
        num_added_toks = self.tokenizer.add_special_tokens(special_tokens_dict)
        
        # self.tokenizer.padding_side = "left"
        
    def __len__(self): # 기본적인 구성
        return len(self.dataset)
    
    def __getitem__(self, idx): # 기본적인 구성
        return self.dataset[idx]
    
    def make_input(self, enc_inputs, dec_inputs):
        enc_input_ids = enc_inputs['input_ids']
        enc_attention_mask = enc_inputs['attention_mask']

        dec_input_ids = dec_inputs['input_ids']        
        real_dec_input_ids = dec_input_ids[:, :-1]
        dec_attention_mask = dec_inputs['attention_mask'][:,:-1]
        
        labels = dec_input_ids[:, 1:].clone()
        
        batch_input = {}
        batch_input['enc_input_ids'] = enc_input_ids
        batch_input['enc_attention_mask'] = enc_attention_mask
        batch_input['dec_input_ids'] = real_dec_input_ids
        batch_input['dec_attention_mask'] = dec_attention_mask
        batch_input['labels'] = labels
        
        return batch_input
    
    def collate_fn(self, batch_data): # 배치를 위한 구성
        '''
            input:
                batch_data: [(query, title), (query, title), ... ]
            return:
                batch_padding_input_token: (B, L) padded
                batch_padding_attention_mask:
        '''
        ftoken, dtoken, rtoken = ['[facet]', '[document]', '[related]']
        
        f_enc, d_enc, r_enc, i_enc = [], [], [], []
        f_dec, d_dec, r_dec, i_dec = [], [], [], []
        for data in batch_data:
            query_string = data['query']
            facet = data['facet']
            
            try:
                document_string = "|".join(data['document'])
            except:
                document_string = ""
                
            try:
                related_string = "|".join(data['related'])
            except:
                related_string = ""
                
            f_string = f"{ftoken} {query_string}"
            d_string = f"{dtoken} {query_string}"
            r_string = f"{rtoken} {query_string}"
            
            f_enc.append(f_string)
            d_enc.append(d_string)
            r_enc.append(r_string)
            
            f_target = "|".join(facet).strip()
            d_target = document_string
            r_target = related_string
            
            f_dec.append(f_target)
            d_dec.append(d_target)
            r_dec.append(r_target)
            
        # 앞뒤로 <s>, </s> 미포함
        f_enc_inputs = self.tokenizer(f_enc, padding=True, truncation=True, max_length=self.tokenizer.model_max_length, return_tensors='pt', add_special_tokens=False)
        d_enc_inputs = self.tokenizer(d_enc, padding=True, truncation=True, max_length=self.tokenizer.model_max_length, return_tensors='pt', add_special_tokens=False)
        r_enc_inputs = self.tokenizer(r_enc, padding=True, truncation=True, max_length=self.tokenizer.model_max_length, return_tensors='pt', add_special_tokens=False)
        
        # 앞뒤로 <s>, </s> 포함
        f_dec_inputs = self.tokenizer(f_dec, padding=True, truncation=True, max_length=self.tokenizer.model_max_length, return_tensors='pt')
        d_dec_inputs = self.tokenizer(d_dec, padding=True, truncation=True, max_length=self.tokenizer.model_max_length, return_tensors='pt')
        r_dec_inputs = self.tokenizer(r_dec, padding=True, truncation=True, max_length=self.tokenizer.model_max_length, return_tensors='pt')
        
        f_input = self.make_input(f_enc_inputs, f_dec_inputs)
        d_input = self.make_input(d_enc_inputs, d_dec_inputs)
        r_input = self.make_input(r_enc_inputs, r_dec_inputs)

        return f_input, d_input, r_input