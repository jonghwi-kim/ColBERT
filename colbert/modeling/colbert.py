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

    def forward(self, Q, D, CS_Q, CS_D, ir_triplet_type='original', seed=12345):
        ### To Do ###
        # Organize the code below. Quite messy :(
        # In particular, the codes for diverse contrastive loss and diverse triplet types.
    
        if self.align_obj is not None:
            query_rep = self.query(*Q)
            doc_rep = self.doc(*D)
        
        if self.align_obj == 'token_contrast':
            contrastive_loss, cs_query_rep, cs_doc_rep = self.compute_token_contrastive_loss(query_rep, doc_rep, CS_Q, CS_D)
        elif self.align_obj == 'colbert_contrast':        
            contrastive_loss, cs_query_rep, cs_doc_rep = self.compute_colbert_contrastive_loss(query_rep, doc_rep, CS_Q, CS_D)
        else:
            contrastive_loss = torch.tensor(0.0)
    
        chance = random.random()
        if (ir_triplet_type =='original') or (ir_triplet_type == 'shuffled' and chance < 0.5):
            if self.align_obj is  None:
                query_rep = self.query(*Q)
                doc_rep = self.doc(*D)
            ir_score = self.max_sim_score(query_rep, doc_rep)
        elif (ir_triplet_type =='codeswitched') or (ir_triplet_type == 'shuffled' and chance >= 0.5):
            if self.align_obj is  None:
                cs_query_rep = self.query(*CS_Q[:2]).repeat(2,1,1)
                cs_doc_rep = self.doc(*CS_D[:2])
            ir_score = self.max_sim_score(cs_query_rep, cs_doc_rep)
        
        return ir_score, contrastive_loss 

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
    
    def compute_colbert_contrastive_loss(self, query_rep, doc_rep, CS_Q, CS_D):
        if CS_Q[0] is None:
            cs_query_rep = query_rep.clone()
            query_colbert_contrastive_loss = 0
            n_cs = 0
        else:
            cs_query_rep = self.query(*CS_Q[:2])
            query_colbert_contrastive_loss = self.colbert_contrastive_loss(query_rep, cs_query_rep)
            cs_query_rep = cs_query_rep.repeat(2,1,1)
            n_cs = 1
        
        if CS_D[0] is None:
            cs_doc_rep = doc_rep.clone()
            doc_colbert_contrastive_loss = 0
        else:
            bsz = int(CS_D[0].size(0)/2)
            cs_doc_rep = self.doc(*CS_D[:2])
            pos_doc_colbert_contrastive_loss = self.colbert_contrastive_loss(doc_rep[:bsz], cs_doc_rep[:bsz])
            neg_doc_colbert_contrastive_loss = self.colbert_contrastive_loss(doc_rep[bsz:], cs_doc_rep[bsz:])
            doc_colbert_contrastive_loss = (pos_doc_colbert_contrastive_loss + neg_doc_colbert_contrastive_loss)/2
            n_cs += 1
            
        colbert_contrastive_loss = (query_colbert_contrastive_loss + doc_colbert_contrastive_loss)/n_cs
        return colbert_contrastive_loss, cs_query_rep, cs_doc_rep
    
    def colbert_contrastive_loss(self, origin, codeswitched):
        
        bsz, seq_len, _ = codeswitched.size()
        
        labels = torch.arange(bsz)
        #labels = torch.cat([torch.arange(bsz) for i in range(2)], dim=0)
        
        labels = (labels.unsqueeze(0) == labels.unsqueeze(1)).float().to(DEVICE)
        #mask = torch.eye(labels.shape[0], dtype=torch.bool).to(DEVICE)
        #labels = labels[~mask].view(labels.shape[0], -1)

        #org_cs = torch.concat((origin[:bsz], codeswitched))
        #max_sim_matrix = (org_cs.unsqueeze(0) @ org_cs.permute(0, 2, 1).unsqueeze(1)).max(3).values.sum(2)
        #breakpoint()
        max_sim_matrix_1 = (origin[:bsz].unsqueeze(0) @ codeswitched.permute(0, 2, 1).unsqueeze(1)).max(3).values.sum(2)/seq_len
        max_sim_matrix_2 = (codeswitched.unsqueeze(0) @ origin[:bsz].permute(0, 2, 1).unsqueeze(1)).max(3).values.sum(2)/seq_len
        #max_sim_matrix = max_sim_matrix[~mask].view(max_sim_matrix.shape[0], -1)

        positives_1 = max_sim_matrix_1[labels.bool()].view(labels.shape[0], -1)
        negatives_1 = max_sim_matrix_1[~labels.bool()].view(max_sim_matrix_1.shape[0], -1)
        logits_1 = torch.cat([positives_1, negatives_1], dim=1)
        
        positives_2 = max_sim_matrix_2[labels.bool()].view(labels.shape[0], -1)
        negatives_2 = max_sim_matrix_2[~labels.bool()].view(max_sim_matrix_2.shape[0], -1)
        logits_2 = torch.cat([positives_2, negatives_2], dim=1)
        
        labels = torch.zeros(logits_1.shape[0], dtype=torch.long).to(DEVICE)
        
        colbert_contrastive_loss = (torch.nn.CrossEntropyLoss()(logits_1, labels) + torch.nn.CrossEntropyLoss()(logits_2, labels))/2

        return colbert_contrastive_loss
    
    def compute_sent_contrastive_loss(self, query_rep, doc_rep, CS_Q, CS_D):
        return
    
    def sent_contrastive_los(self):
        return
    
    def compute_token_contrastive_loss(self, query_rep, doc_rep, CS_Q, CS_D):
        
        if CS_Q[0] is None:
            cs_query_rep = query_rep.clone()
            query_token_contrastive_loss = 0
            n_cs = 0
        else:
            cs_query_rep = self.query(*CS_Q[:2])
            query_token_contrastive_loss = self.token_contrastive_loss(query_rep, cs_query_rep, cs_position=CS_Q[2])
            cs_query_rep = cs_query_rep.repeat(2,1,1)
            n_cs = 1
        
        if CS_D[0] is None:
            cs_doc_rep = doc_rep.clone()
            doc_token_contrastive_loss = 0
        else:
            cs_doc_rep = self.doc(*CS_D[:2])
            doc_token_contrastive_loss = self.token_contrastive_loss(doc_rep, cs_doc_rep, cs_position=CS_D[2])
            n_cs += 1
            
        token_contrastive_loss = (query_token_contrastive_loss + doc_token_contrastive_loss)/n_cs
        return token_contrastive_loss, cs_query_rep, cs_doc_rep
    
    def token_contrastive_loss(self, origin, codeswitched, cs_position):
        bsz, seqlen, _ = codeswitched.size()
        token_distance_matrix = self.distance_matrix(origin[:bsz], codeswitched)
        logprobs = F.log_softmax(token_distance_matrix.view(-1, seqlen), dim=-1)
        gold = torch.arange(seqlen).view(-1,).expand(bsz, seqlen).contiguous().view(-1).cuda(token_distance_matrix.get_device())
        token_contrastive_loss = -logprobs.gather(dim=-1, index=gold.unsqueeze(1)).squeeze(1)
        token_contrastive_loss = token_contrastive_loss.view(bsz, seqlen) * cs_position.cuda(token_distance_matrix.get_device())
        token_contrastive_loss = torch.sum(token_contrastive_loss) / cs_position.sum()
        return token_contrastive_loss
    
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
