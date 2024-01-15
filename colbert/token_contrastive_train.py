import os
import random
import torch
import copy

import colbert.utils.distributed as distributed

from colbert.utils.parser import Arguments
from colbert.utils.runs import Run
from colbert.training.training import train
from colbert.utils.language_code import short2long, long2short


def main():
    parser = Arguments(description='Training ColBERT with <query, positive passage, negative passage> triples.')

    parser.add_model_parameters()
    parser.add_model_training_parameters()
    parser.add_training_input()
    
    parser.add_argument("--align_obj", action="store_true")
    parser.add_argument("--alpha", type=float, default=0.5, help="Weight for the alignment loss.")
    parser.add_argument("--using_lexicon", action="store_true")
    parser.add_argument("--target_query_lang", default="en", help="Target Query Code-Switched Language (EN -> Target Lang)" )
    parser.add_argument("--target_doc_lang", default="en", help="Target Document Code-Switched Language (EN -> Target Lang)" )
    parser.add_argument("--lexicon", help="Folder containing bilingual lexicon files (run extract_bilingual_dictionary_from_mPLM_vocab.py first)")
    parser.add_argument("--switching_prob", default=0.5, type=float, help="Probability of switching language in a given query or document.")
    parser.add_argument("--seed", '--s', default=12345, type=int)
    #parser.add_argument("--codeswitched_triplet_s_dir", help="Output directory for writing code-switched training data.", required=True)
    args = parser.parse()

    args.target_query_lang = long2short[args.target_query_lang]
    args.target_doc_lang = long2short[args.target_doc_lang]

    assert args.bsize % args.accumsteps == 0, ((args.bsize, args.accumsteps),
                                               "The batch size must be divisible by the number of gradient accumulation steps.")
    assert args.query_maxlen <= 512
    assert args.doc_maxlen <= 512

    args.lazy = args.collection is not None

    with Run.context(consider_failed_if_interrupted=False):
        train(args)


if __name__ == "__main__":
    main()
