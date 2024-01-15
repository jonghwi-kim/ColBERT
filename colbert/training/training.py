import os
import random
import time
import torch
import torch.nn as nn
import numpy as np

from transformers import AdamW
from colbert.utils.runs import Run
from colbert.utils.amp import MixedPrecisionManager

from colbert.training.lazy_batcher import LazyBatcher
from colbert.training.eager_batcher import EagerBatcher
from colbert.parameters import DEVICE

from colbert.modeling.colbert import ColBERT
from colbert.utils.utils import print_message
from colbert.training.utils import print_progress, manage_checkpoints


def train(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.distributed:
        torch.cuda.manual_seed_all(args.seed)

    if args.distributed:
        assert args.bsize % args.nranks == 0, (args.bsize, args.nranks)
        assert args.accumsteps == 1
        args.bsize = args.bsize // args.nranks

        print("Using args.bsize =", args.bsize, "(per process) and args.accumsteps =", args.accumsteps)

    if args.lazy:
        reader = LazyBatcher(args, (0 if args.rank == -1 else args.rank), args.nranks)
    else:
        reader = EagerBatcher(args, (0 if args.rank == -1 else args.rank), args.nranks)

    if args.rank not in [-1, 0]:
        torch.distributed.barrier()

    colbert = ColBERT.from_pretrained(args.base_model,
                                      query_maxlen=args.query_maxlen,
                                      doc_maxlen=args.doc_maxlen,
                                      dim=args.dim,
                                      similarity_metric=args.similarity,
                                      mask_punctuation=args.mask_punctuation,
                                      align_obj=args.align_obj)
    colbert.roberta.resize_token_embeddings(len(colbert.tokenizer))

    if args.checkpoint is not None:
        assert args.resume_optimizer is False, "TODO: This would mean reload optimizer too."
        print_message(f"#> Starting from checkpoint {args.checkpoint} -- but NOT the optimizer!")

        checkpoint = torch.load(args.checkpoint, map_location='cpu')

        try:
            colbert.load_state_dict(checkpoint['model_state_dict'])
        except:
            print_message("[WARNING] Loading checkpoint with strict=False")
            colbert.load_state_dict(checkpoint['model_state_dict'], strict=False)

    if args.rank == 0:
        torch.distributed.barrier()

    colbert = colbert.to(DEVICE)
    colbert.train()

    if args.distributed:
        colbert = torch.nn.parallel.DistributedDataParallel(colbert, device_ids=[args.rank],
                                                            output_device=args.rank,
                                                            find_unused_parameters=True)

    optimizer = AdamW(filter(lambda p: p.requires_grad, colbert.parameters()), lr=args.lr, eps=1e-8)
    optimizer.zero_grad()

    amp = MixedPrecisionManager(args.amp)
    criterion = nn.CrossEntropyLoss()
    labels = torch.zeros(args.bsize, dtype=torch.long, device=DEVICE)

    start_time = time.time()
    ir_train_loss = 0.0
    ta_train_loss = 0.0
    start_batch_idx = 0
    alpha = args.alpha

    if args.resume: 
        assert args.checkpoint is not None
        start_batch_idx = checkpoint['batch']

        reader.skip_to_batch(start_batch_idx, checkpoint['arguments']['bsize'])

    for batch_idx, BatchSteps in zip(range(start_batch_idx, args.maxsteps), reader):
        this_batch_ir_loss = 0.0
        this_batch_ta_loss = 0.0

        for en_queries, en_passages, cs_queries, cs_docs in BatchSteps:
            with amp.context():
                ir_scores, token_alignment_loss = colbert(en_queries, en_passages, cs_queries, cs_docs)
                ir_scores = ir_scores.view(2, -1).permute(1, 0)
                ir_loss = criterion(ir_scores, labels[:ir_scores.size(0)])
                ir_loss = ir_loss / args.accumsteps
                
                token_alignment_loss = token_alignment_loss / args.accumsteps
                #total_loss = alpha*ir_loss + (1-alpha)*token_alignment_loss
                #ir_loss = alpha*ir_loss
                #token_alignment_loss = (1-alpha)*token_alignment_loss
                total_loss = ir_loss + token_alignment_loss
                
            if args.rank < 1 and batch_idx % 100 == 0:
                print_progress(ir_scores)

            amp.backward(total_loss)

            ir_train_loss += ir_loss.item()
            this_batch_ir_loss += ir_loss.item()
            
            ta_train_loss += token_alignment_loss.item()
            this_batch_ta_loss += token_alignment_loss.item()

        amp.step(colbert, optimizer)

        if args.rank < 1:
            avg_ir_loss = ir_train_loss / (batch_idx+1)
            avg_ta_loss = ta_train_loss / (batch_idx+1)

            num_examples_seen = (batch_idx - start_batch_idx) * args.bsize * args.nranks
            elapsed = float(time.time() - start_time)

            log_to_mlflow = (batch_idx % 20 == 0)
            Run.log_metric('train/avg_total_loss', (avg_ir_loss+avg_ta_loss), step=batch_idx, log_to_mlflow=log_to_mlflow)
            Run.log_metric('train/avg_ir_loss', avg_ir_loss, step=batch_idx, log_to_mlflow=log_to_mlflow)
            Run.log_metric('train/avg_ta_loss', avg_ta_loss, step=batch_idx, log_to_mlflow=log_to_mlflow)
            
            Run.log_metric('train/batch_total_loss', (this_batch_ir_loss+this_batch_ta_loss), step=batch_idx, log_to_mlflow=log_to_mlflow)
            Run.log_metric('train/batch_ir_loss', this_batch_ir_loss, step=batch_idx, log_to_mlflow=log_to_mlflow)
            Run.log_metric('train/batch_ta_loss', this_batch_ta_loss, step=batch_idx, log_to_mlflow=log_to_mlflow)
            
            Run.log_metric('train/examples', num_examples_seen, step=batch_idx, log_to_mlflow=log_to_mlflow)
            Run.log_metric('train/throughput', num_examples_seen / elapsed, step=batch_idx, log_to_mlflow=log_to_mlflow)

            manage_checkpoints(args, colbert, optimizer, batch_idx+1)
            if batch_idx % 100 == 0:
                print_message(batch_idx+1, (avg_ir_loss+avg_ta_loss), avg_ir_loss, avg_ta_loss)
