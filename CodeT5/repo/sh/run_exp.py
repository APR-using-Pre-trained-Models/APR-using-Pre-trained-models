#!/usr/bin/env python
import os
import argparse


def get_cmd(task, sub_task, model_tag, gpu, data_num, bs, lr, source_length, target_length, patience, epoch, warmup,
            model_dir, summary_dir, res_fn, max_steps=None, save_steps=None, log_steps=None, nbest=1, load_model_path=None, do_train="store_true"):
    if max_steps is None:
        cmd_str = 'bash exp_with_args.sh %s %s %s %d %d %d %d %d %d %d %d %d %s %s %s %d %s %s' % \
                  (task, sub_task, model_tag, gpu, data_num, bs, lr, source_length, target_length, patience, epoch,
                   warmup, model_dir, summary_dir, res_fn, nbest, load_model_path, do_train)
    else:
        cmd_str = 'bash exp_with_args.sh %s %s %s %d %d %d %d %d %d %d %d %d %s %s %s %d %d %d %d %s %s' % \
                  (task, sub_task, model_tag, gpu, data_num, bs, lr, source_length, target_length, patience, epoch,
                   warmup, model_dir, summary_dir, res_fn, max_steps, save_steps, log_steps, nbest, load_model_path, do_train)
    return cmd_str


def get_args_by_task_model(task, sub_task, model_tag):
    if task == 'translate':
        # java-cs: Read 10300 examples, avg src len: 13, avg trg len: 15, max src len: 136, max trg len: 118
        # [TOKENIZE] avg src len: 45, avg trg len: 56, max src len: 391, max trg len: 404
        src_len = 320
        trg_len = 256
        epoch = 100
        patience = 5
    elif task == 'summarize':
        # ruby: Read 24927 examples, avg src len: 66, avg trg len: 12, max src len: 501, max trg len: 146
        # [TOKENIZE] avg src len: 100, avg trg len: 13, max src len: 1250, max trg len: 161
        # Python: Read 251820 examples, avg src len: 100, avg trg len: 11, max src len: 512, max trg len: 222
        # [TOKENIZE] avg src len: 142, avg trg len: 12, max src len: 2016, max trg len: 245
        # Javascript: Read 58025 examples, avg src len: 114, avg trg len: 11, max src len: 512, max trg len: 165
        # [TOKENIZE] avg src len: 136, avg trg len: 12, max src len: 3016, max trg len: 177
        src_len = 256
        trg_len = 128
        epoch = 15
        patience = 2
    elif task == 'refine':
        # small: Read 46680 examples, avg src len: 31, avg trg len: 28, max src len: 50, max trg len: 50
        # [TOKENIZE] avg src len: 50, avg trg len: 45, max src len: 129, max trg len: 121
        # medium:  Read 52364 examples, avg src len: 74, avg trg len: 73, max src len: 100, max trg len: 100
        # [TOKENIZE] avg src len: 117, avg trg len: 114, max src len: 238, max trg len: 238
        if sub_task == 'small':
            src_len = 130
            trg_len = 120
        elif sub_task == 'medium':
            src_len = 240
            trg_len = 240
        epoch = 50
        patience = 5
    elif task == 'refine_R4R':
        #  Read 53198 examples, avg src len: 98, avg trg len: 7, max src len: 339, max trg len: 45 
        # [TOKENIZE] avg src len: 250, avg trg len: 17, max src len: 561, max trg len: 116
        if sub_task == 'cc':
            src_len = 512
            trg_len = 120
        elif sub_task == 'c':
            src_len = 496
            trg_len = 120
        epoch = 10
        patience = 5
    elif task == 'refine_tufano':
        # Read 13756 examples, avg src len: 69, avg trg len: 48, max src len: 315, max trg len: 100
        # [TOKENIZE] avg src len: 111, avg trg len: 70, max src len: 590, max trg len: 194  
        if sub_task == 'cc':
            src_len = 512
            trg_len = 200
        elif sub_task == 'c':
            src_len = 200
            trg_len = 200
        epoch = 10
        patience = 5
    elif task == 'concode':
        # Read 100000 examples, avg src len: 71, avg trg len: 26, max src len: 567, max trg len: 140
        # [TOKENIZE] avg src len: 213, avg trg len: 33, max src len: 2246, max trg len: 264
        src_len = 320
        trg_len = 150
        epoch = 30
        patience = 3
    elif task == 'defect':
        # Read 21854 examples, avg src len: 187, avg trg len: 1, max src len: 12195, max trg len: 1
        # [TOKENIZE] avg src len: 597, avg trg len: 1, max src len: 41447, max trg len: 1
        src_len = 512
        trg_len = 3
        epoch = 10
        patience = 2
    elif task == 'clone':
        # Read 901028 examples, avg src len: 120, avg trg len: 123, max src len: 5270, max trg len: 5270
        # [TOKENIZE] avg src len: 318, avg trg len: 323, max src len: 15111, max trg len: 15111
        src_len = 400
        trg_len = 400
        epoch = 1
        patience = 2

    if 'codet5_small' in model_tag:
        bs = 32
        if task == 'summarize' or task == 'translate' or (task == 'refine' and sub_task == 'small'):
            bs = 64
        elif task == 'clone':
            bs = 25
    else:
        # change
        #bs = 32
        bs = 4
        if task == 'translate':
            bs = 25
        elif task == 'summarize':
            bs = 48
        elif task == 'clone':
            if model_tag in ['codebert', 'roberta']:
                bs = 16
            else:
                bs = 10
    lr = 5
    if task == 'concode':
        lr = 10
    elif task == 'defect':
        lr = 2
    return bs, lr, src_len, trg_len, patience, epoch


def run_one_exp(args):
    bs, lr, src_len, trg_len, patience, epoch = get_args_by_task_model(args.task, args.sub_task, args.model_tag)
    print('============================Start Running==========================')
    cmd_str = get_cmd(task=args.task, sub_task=args.sub_task, model_tag=args.model_tag, gpu=args.gpu,
                      data_num=args.data_num, bs=bs, lr=lr, source_length=src_len, target_length=trg_len,
                      patience=patience, epoch=epoch, warmup=1000,
                      model_dir=args.model_dir, summary_dir=args.summary_dir,
                      res_fn='{}/{}_{}.txt'.format(args.res_dir, args.task, args.model_tag),
                      nbest=args.nbest, load_model_path=args.load_model_path, do_train=args.do_train)
    print('%s\n' % cmd_str)
    os.system(cmd_str)


def run_multi_task_exp(args):
    # Total train data num = 1149722 (for all five tasks)
    if 'codet5_small' in args.model_tag:
        bs, lr, max_steps, save_steps, log_steps = 60, 5, 600000, 20000, 100
    else:
        bs, lr, max_steps, save_steps, log_steps = 25, 5, 800000, 20000, 100

    if args.data_num != -1:
        max_steps, save_steps, log_steps = 1000, 200, 50
    print('============================Start Running==========================')
    cmd_str = get_cmd(task='multi_task', sub_task='none', model_tag=args.model_tag, gpu=args.gpu,
                      data_num=args.data_num, bs=bs, lr=lr, source_length=-1, target_length=-1,
                      patience=-1, epoch=-1, warmup=1000,
                      model_dir=args.model_dir, summary_dir=args.summary_dir,
                      res_fn='{}/multi_task_{}.txt'.format(args.res_dir, args.model_tag),
                      max_steps=max_steps, save_steps=save_steps, log_steps=log_steps)
    print('%s\n' % cmd_str)
    os.system(cmd_str)


def get_sub_tasks(task):
    if task == 'summarize':
        sub_tasks = ['ruby', 'javascript', 'go', 'python', 'java', 'php']
    elif task == 'translate':
        sub_tasks = ['java-cs', 'cs-java']
    elif task == 'refine':
        sub_tasks = ['small', 'medium']
    elif task in ['refine_R4R', 'refine_tufano']:
        sub_tasks = ['cc', 'c']
    elif task in ['concode', 'defect', 'clone', 'multi_task']:
        sub_tasks = ['none']
    return sub_tasks


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_tag", type=str, default='codet5_base',
                        choices=['roberta', 'codebert', 'bart_base', 'codet5_small', 'codet5_base'])
    parser.add_argument("--task", type=str, default='summarize', choices=['summarize', 'concode', 'translate',
                                                                          'refine', 'refine_R4R', 'refine_tufano', 'defect', 'clone', 'multi_task'])
    parser.add_argument("--sub_task", type=str, default='ruby')
    parser.add_argument("--res_dir", type=str, default='results', help='directory to save fine-tuning results')
    parser.add_argument("--model_dir", type=str, default='saved_models', help='directory to save fine-tuned models')
    parser.add_argument("--summary_dir", type=str, default='tensorboard', help='directory to save tensorboard summary')
    parser.add_argument("--data_num", type=int, default=-1, help='number of data instances to use, -1 for full data')
    parser.add_argument("--gpu", type=int, default=0, help='index of the gpu to use in a cluster')
    parser.add_argument("--nbest", type=int, default=1, help='to generate n predictions')
    parser.add_argument("--load_model_path", type=str, default=None, help='give model path here')
    parser.add_argument("--do_train", type=str, default="store_true", help='determines to train or not. For direct testing it should be store_false.')
    
    args = parser.parse_args()

    if not os.path.exists(args.res_dir):
        os.makedirs(args.res_dir)

    assert args.sub_task in get_sub_tasks(args.task)
    if args.task != 'multi_task':
        run_one_exp(args)
    else:
        run_multi_task_exp(args)