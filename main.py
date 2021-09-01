import argparse
import os
from data.classes import get_voc_class

if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    # Environment, Dataset
    parser.add_argument("--num_workers", default=os.cpu_count()//2, type=int)
    parser.add_argument("--dataset", default="voc12", choices=['voc12', 'cityscapes'], type=str, 
                        help="Choose the dataset which to train.")
    # VOC
    parser.add_argument("--voc12_root", default="../../dataset/VOC/", type=str,
                        help="Path to VOC 2012 Devkit, must contain ./JPEGImages as subdirectory.")
    # Cityscapes
    parser.add_argument("--cityscapes_root", default="../../dataset/cityscapes", type=str,
                        help="Path to Cityscapes dataset, must contain ./leftImg8bit or gtFine or gtCoarse are located.")
    parser.add_argument("--cityscapes_mode", default="fine", type=str, 
                        help="fine or coarse")
    parser.add_argument("--eval_set", default="train", type=str,
                        help="voc12: train/test/val, cityscapes: train/train_extra(coarse mode)/val/test(fine mode)")

    # Output Path
    #parser.add_argument("--log_name", default="sample_train_eval", type=str)
    parser.add_argument("--weights_dir", default="result/weights", type=str)
    parser.add_argument("--cam_out_dir", default="result/cam", type=str)
    #parser.add_argument("--sem_seg_out_dir", default="result/sem_seg", type=str)

    # Finetuning
    parser.add_argument("--seed", default=42, type=int)
    parser.add_argument("--network", default="resnet50", type=str)
    parser.add_argument("--crop_size", default=224, type=int)
    parser.add_argument("--batch_size", default=32, type=int)
    parser.add_argument("--epoches", default=15, type=int)
    parser.add_argument("--learning_rate", default=0.1, type=float)
    parser.add_argument("--weight_decay", default=1e-4, type=float)
    parser.add_argument("--eval_thres_start", default=0.05, type=float)
    parser.add_argument("--eval_thres_limit", default=1., type=float)
    parser.add_argument("--eval_thres_jump", default=0.05, type=float)
    #parser.add_argument("--cam_scales", default=(1.0, 0.5, 1.5, 2.0),
    #                    help="Multi-scale inferences") 

    # Step
    parser.add_argument("--finetune_skip", action="store_true")
    parser.add_argument("--gen_cam_skip", action="store_true")
    parser.add_argument("--eval_cam_skip", action="store_true")
    #parser.add_argument("--save_cam", action="store_true",
    #                    help="The flag which determines save cam before evaluation.")

    args = parser.parse_args()

    # Run
    args.voc_class = get_voc_class()
    args.voc_class_num = len(args.voc_class)
    # finetuning
    if args.finetune_skip is not True:
        import finetune
        finetune.run(args)
    # generate cam
    if args.gen_cam_skip is not True:
        import gen_cam
        gen_cam.run(args)
    # evaluate cam
    if args.eval_cam_skip is not True:
        import eval_cam
        eval_cam.run(args)
