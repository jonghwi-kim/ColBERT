import os
import ujson

from functools import partial
from colbert.utils.utils import print_message
from colbert.modeling.tokenization import QueryTokenizer, DocTokenizer, tensorize_triples

from colbert.utils.runs import Run


class LazyBatcher():
    def __init__(self, args, rank=0, nranks=1):
        self.bsize, self.accumsteps = args.bsize, args.accumsteps

        self.query_tokenizer = QueryTokenizer(args.query_maxlen)
        self.doc_tokenizer = DocTokenizer(args.doc_maxlen)
        self.tensorize_triples = partial(tensorize_triples, self.query_tokenizer, self.doc_tokenizer)
        self.position = 0

        self.triples = self._load_triples(args.triples, rank, nranks)
        self.queries = self._load_queries(args.queries)
        self.collection = self._load_collection(args.collection)
        
        if args.using_lexicon:
            self.lexicons = self._load_lexicon(args.lexicon, args.target_query_lang, args.target_doc_lang)
            
    def _load_lexicon(self, lexicon_dir, query_lang, doc_lang):
        
        # Now, lexicon is a single bilingual dictionary 
        # (e.g., {en -> fr}, {en -> de}, {en -> es}, en -> it}, {en -> ru}, {en -> zh}, ... )
        print_message("#> Loading lexicons...")
        
        assert query_lang != 'en' or doc_lang != 'en', "You should use at least one bilingual lexicon (e.g. en -> non-en)"
        
        lexicons = {}
        if query_lang != 'en': lexicons['lexicon_for_query'] = ujson.load(open(os.path.join(lexicon_dir, f"en_to_{query_lang}_token_dict_csls.json")))
        if doc_lang != 'en': lexicons['lexicon_for_doc'] = ujson.load(open(os.path.join(lexicon_dir, f"en_to_{doc_lang}_token_dict_csls.json")))

        # To Do : implementation of using multiple lexicons 
        # (e.g., en -> {fr, de, es, it, ru, zh, ...} )
        
        #for filename in os.listdir(lexicon_dir):            
        #     with open(os.path.join(lexicon_dir, filename)) as f:
        #         for line in f:
        #             word, idx = line.strip().split('\t')
        #             idx = int(idx)
        #             lexicon[word] = idx
                    
        return lexicons

    def _load_triples(self, path, rank, nranks):
        """
        NOTE: For distributed sampling, this isn't equivalent to perfectly uniform sampling.
        In particular, each subset is perfectly represented in every batch! However, since we never
        repeat passes over the data, we never repeat any particular triple, and the split across
        nodes is random (since the underlying file is pre-shuffled), there's no concern here.
        """
        print_message("#> Loading triples...")

        triples = []

        with open(path) as f:
            for line_idx, line in enumerate(f):
                if line_idx % nranks == rank:
                    qid, pos, neg = line.split('\t')
                    #ujson.loads(line)
                    triples.append((qid, pos, neg))

        return triples

    def _load_queries(self, path):
        print_message("#> Loading queries...")

        queries = {}

        with open(path) as f:
            for line in f:
                qid, query = line.strip().split('\t')
                qid = int(qid)
                queries[qid] = query

        return queries

    def _load_collection(self, path):
        print_message("#> Loading collection...")

        collection = []

        with open(path) as f:
            for line_idx, line in enumerate(f):
                pid, passage, *_ = line.strip().split('\t')
                #pid, passage, title, *_ = line.strip().split('\t')
                assert pid == 'id' or int(pid) == line_idx

                #passage = title + ' | ' + passage
                collection.append(passage)

        return collection

    def __iter__(self):
        return self

    def __len__(self):
        return len(self.triples)

    def __next__(self):
        offset, endpos = self.position, min(self.position + self.bsize, len(self.triples))
        self.position = endpos

        if offset + self.bsize > len(self.triples):
            raise StopIteration

        queries, positives, negatives = [], [], []

        for position in range(offset, endpos):
            query, pos, neg = self.triples[position]
            query, pos, neg = self.queries[int(query)], self.collection[int(pos)], self.collection[int(neg)]

            queries.append(query)
            positives.append(pos)
            negatives.append(neg)

        return self.collate(queries, positives, negatives)

    def collate(self, queries, positives, negatives):
        assert len(queries) == len(positives) == len(negatives) == self.bsize

        return self.tensorize_triples(queries, positives, negatives, self.bsize // self.accumsteps, self.lexicons)

    def skip_to_batch(self, batch_idx, intended_batch_size):
        Run.warn(f'Skipping to batch #{batch_idx} (with intended_batch_size = {intended_batch_size}) for training.')
        self.position = intended_batch_size * batch_idx
