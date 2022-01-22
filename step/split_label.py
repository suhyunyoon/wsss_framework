import os
import random

def run(args):
    random.seed(args.seed)

    with open(args.train_list, 'r') as f:
        files = f.read().split('\n')[:-1]
    
    num_data = len(files)
    
    split_data = random.sample(files, int(num_data * args.labeled_ratio))

    new_list = '\n'.join(split_data)

    new_file = os.path.join(os.path.dirname(args.train_list), 'new_' + os.path.basename(args.train_list))
    with open(new_file, 'w') as f:
        f.write(new_list)
    
    # add new list
    if args.use_unlabeled:
        args.unlabeled_train_list = new_file
    else:
        args.train_list = new_file


if __name__=='__main__':
    import argparse
    parser = argparse.ArgumentParser()

    # Environment
    parser.add_argument("--seed", default=42, type=int)

    # VOC12 Dataset
    parser.add_argument("--train_list", default="voc12/train_aug.txt", type=str)

    # Step
    parser.add_argument("--labeled_ratio", default=1., type=float)
    parser.add_argument("--use_unlabeled", action="store_true", help="Use unlabeled images after train_cam")

    args = parser.parse_args()
    
    run(args)
