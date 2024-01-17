import random
import string
import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers import RobertaPreTrainedModel, XLMRobertaConfig, XLMRobertaTokenizer, XLMRobertaModel
from colbert.parameters import DEVICE


class ColBERT(RobertaPreTrainedModel):
    
    config_class = XLMRobertaConfig
    
    def __init__(self, config, query_maxlen, doc_maxlen, mask_punctuation, dim=128, similarity_metric='cosine', align_obj=None):

        super(ColBERT, self).__init__(config)
        print(config)

        self.query_maxlen = query_maxlen
        self.doc_maxlen = doc_maxlen
        self.similarity_metric = similarity_metric
        self.dim = dim
        self.align_obj = align_obj

        self.mask_punctuation = mask_punctuation
        self.skiplist = {}
        
        self.tokenizer = XLMRobertaTokenizer.from_pretrained('xlm-roberta-base')
        self.tokenizer.add_tokens(['[unused1]'])
        self.tokenizer.add_tokens(['[unused2]'])
        
        self.roberta = XLMRobertaModel(config)

        if self.mask_punctuation:    
            self.skiplist = {w: True
                             for symbol in string.punctuation
                             for w in [symbol, self.tokenizer.encode(symbol, add_special_tokens=False)[0]]}

        self.linear = nn.Linear(config.hidden_size, dim, bias=False)
        self.init_weights()

    def forward(self, Q, D, CS_Q, CS_D, ir_triplet_type='original'):
        ### To Do ###
        # Reduce the length of forward pass codes 
        # In particular, the codes for computing alignment loss
        
        query_rep = self.query(*Q)
        doc_rep = self.doc(*D)
        
        if self.align_obj:
            if CS_Q[0] is None:
                cs_query_rep = query_rep.clone()
                query_token_alignment_loss = 0
                n_cs = 0
            else:
                cs_query_rep = self.query(*CS_Q[:2])
                query_token_alignment_loss = self.compute_alignment_loss(query_rep, cs_query_rep, cs_position=CS_Q[2])
                cs_query_rep = cs_query_rep.repeat(2,1,1)
                n_cs = 1
                
            if CS_D[0] is None:
                cs_doc_rep = doc_rep.clone()
                doc_token_alignment_loss = 0
            else:
                cs_doc_rep = self.doc(*CS_D[:2])
                doc_token_alignment_loss = self.compute_alignment_loss(doc_rep, cs_doc_rep, cs_position=CS_D[2])
                n_cs += 1
                
            if ir_triplet_type =='original':
                ir_score = self.max_sim_score(query_rep, cs_doc_rep)
            elif ir_triplet_type == 'shuffled' and random.random() < 0.5:
                ir_score = self.max_sim_score(query_rep, cs_doc_rep)
            else:
                ir_score = self.max_sim_score(cs_query_rep, cs_doc_rep)
            
            token_alignment_loss = (query_token_alignment_loss + doc_token_alignment_loss)/n_cs
            return ir_score, token_alignment_loss 
        
        return self.max_sim_score(query_rep, doc_rep)

    def query(self, input_ids, attention_mask):
        input_ids, attention_mask = input_ids.to(DEVICE), attention_mask.to(DEVICE)
        Q = self.roberta(input_ids, attention_mask=attention_mask)[0]
        Q = self.linear(Q)

        return torch.nn.functional.normalize(Q, p=2, dim=2)

    def doc(self, input_ids, attention_mask, keep_dims=True):
        input_ids, attention_mask = input_ids.to(DEVICE), attention_mask.to(DEVICE)
        D = self.roberta(input_ids, attention_mask=attention_mask)[0]
        D = self.linear(D)

        mask = torch.tensor(self.mask(input_ids), device=DEVICE).unsqueeze(2).float()
        D = D * mask

        D = torch.nn.functional.normalize(D, p=2, dim=2)

        if not keep_dims:
            D, mask = D.cpu().to(dtype=torch.float16), mask.cpu().bool().squeeze(-1)
            D = [d[mask[idx]] for idx, d in enumerate(D)]
        return D
    
    def compute_alignment_loss(self, origin, codeswitched, cs_position):
        bsz, seqlen, _ = codeswitched.size()
        token_distance_matrix = self.distance_matrix(origin[:bsz], codeswitched)
        logprobs = F.log_softmax(token_distance_matrix.view(-1, seqlen), dim=-1)
        gold = torch.arange(seqlen).view(-1,).expand(bsz, seqlen).contiguous().view(-1).cuda(token_distance_matrix.get_device())
        token_alignment_loss = -logprobs.gather(dim=-1, index=gold.unsqueeze(1)).squeeze(1)
        token_alignment_loss = token_alignment_loss.view(bsz, seqlen) * cs_position.cuda(token_distance_matrix.get_device())
        token_alignment_loss = torch.sum(token_alignment_loss) / cs_position.sum()
        return token_alignment_loss
    
    def distance_matrix(self, origin, codeswitched):
        if self.similarity_metric =='cosine':
            return torch.matmul(origin, codeswitched.transpose(1,2))
        elif self.similarity_metric =='l2':
            return -1.0*((origin.unsqueeze(2) - codeswitched.unsqueeze(1))**2).sum(-1)

    def max_sim_score(self, Q, D):
        if self.similarity_metric == 'cosine':
            return (Q @ D.permute(0, 2, 1)).max(2).values.sum(1)

        assert self.similarity_metric == 'l2'
        return (-1.0 * ((Q.unsqueeze(2) - D.unsqueeze(1))**2).sum(-1)).max(-1).values.sum(-1)

    def mask(self, input_ids):
        # For masking CLS token (<s>)
        #mask = [[(x not in self.skiplist) and (x != 0) for x in d] for d in input_ids.cpu().tolist()]
        mask = [[(x not in self.skiplist) and (x != 1) for x in d] for d in input_ids.cpu().tolist()]
        return mask
