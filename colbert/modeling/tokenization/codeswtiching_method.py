import random
import torch
import numpy as np

from nltk.tokenize import TreebankWordTokenizer, TreebankWordDetokenizer
from nltk.corpus import stopwords

count_all_words = False
en_stopwords = None if count_all_words else set(stopwords.words('english'))

def mplm_dictionary_codeswitch(tokenizer, token_ids, lexicon, switching_prob=0.5):
    code_switched_token_ids = token_ids.clone()
    cs_index = torch.zeros(token_ids.size(0), token_ids.size(1))
    
    token_count_list = []
    cs_count_list = []
    for query_idx in range(token_ids.size(0)):
        token_count = 0
        cs_count = 0
        for token_idx in range(token_ids.size(1)):
            token_id = code_switched_token_ids[query_idx, token_idx].item()
            if token_id in (tokenizer.tok.all_special_ids + [5, 250002, 250003]):
                continue
            token_count += 1
            if str(token_id) in lexicon and random.random() < switching_prob:
                cs_count+=1
                code_switched_token_ids[query_idx, token_idx] = lexicon[str(token_id)][0]
                cs_index[query_idx, token_idx] = 1
        token_count_list.append(token_count)
        cs_count_list.append(cs_count)
    
    stat = {'num_token': np.mean(token_count_list), 'num_cs_token': np.mean(cs_count_list)}
    return code_switched_token_ids, cs_index, stat


def golden_dictionary_codeswitch(sentences, lexicons, word_tokenizer, word_detokenizer, switching_prob=0.5):
    
    assert lexicons is not None, "No lexicons provided. Check whether you are using the correct lexicon file."
    
    #word_tokenizer = TreebankWordTokenizer()
    #word_detokenizer = TreebankWordDetokenizer()
    
    code_switched_sentences = []
    n_word_list = []
    cs_count_list = []
    for sentence in sentences:
        tokens = word_tokenizer.tokenize(sentence)
        n_word_list.append(len(tokens))
        
        code_switched = []
        cs_count = 0
        for token in tokens:
            is_content_word = token not in en_stopwords
            if random.random() < switching_prob and token.lower() in lexicons:
                code_switched_token = random.choice(lexicons[token.lower()])
                code_switched.append(code_switched_token)
                cs_count += 1 if is_content_word or count_all_words else 0
            else:
                code_switched.append(token)
                
        cs_count_list.append(cs_count)
        code_switched = word_detokenizer.detokenize(code_switched)
        code_switched_sentences.append(code_switched)
        
    return code_switched_sentences, {'num_token': np.mean(n_word_list), 'num_cs_token': np.mean(cs_count_list)}
        
        
