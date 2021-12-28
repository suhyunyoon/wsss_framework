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
    # set
    parser.add_argument("--train_set", default="train_aug", type=str)
    parser.add_argument("--eval_set", default="val", type=str,
                        help="voc12: train/val/trainval/train_aug, cityscapes: train/train_extra(coarse mode)/val/test(fine mode)")

    # Output Path
    #parser.add_argument("--log_name", default="sample_train_eval", type=str)
    parser.add_argument("--weights_dir", default="result/weights", type=str)
    parser.add_argument("--cam_out_dir", default="result/cam", type=str)
    #parser.add_argument("--sem_seg_out_dir", default="result/sem_seg", type=str)

    # Finetuning
    parser.add_argument("--seed", default=42, type=int)
    parser.add_argument("--network", default="resnet50", type=str)
    '''
                         choices=['resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152', 
                         'resnext50_32x4d', 'resnext101_32x8d', 'wide_resnet50_2', 'wide_resnet101_2',
                         'vgg11', 'vgg13', 'vgg16', 'vgg19', 'vgg11_bn', 'vgg13_bn', 'vgg16_bn', 'vgg19_bn',
                         'dino_resnet50', 'dino_vits16', 'dino_vits8', 'dino_vitb16', 'dino_vitb8',
                         'dino_xcit_small_12_p16', 'dino_xcit_small_12_p8', 'dino_xcit_medium_24_p16', 'dino_xcit_medium_24_p8',
                         'irn.net.resnet50_cam', 'irn.net.resnet50_irn'])
    '''
    parser.add_argument("--crop_size", default=224, type=int)
    parser.add_argument("--batch_size", default=64, type=int)
    parser.add_argument("--epochs", default=100, type=int)
    parser.add_argument("--learning_rate", default=0.1, type=float)
    parser.add_argument("--weight_decay", default=1e-4, type=float)
    # percent(0~100%)
    parser.add_argument("--eval_thres_start", default=5, type=float)
    parser.add_argument("--eval_thres_limit", default=100, type=float)
    parser.add_argument("--eval_thres_jump", default=5, type=float)
    parser.add_argument("--verbose_interval", default=5, type=int)
    #parser.add_argument("--cam_scales", default=(1.0, 0.5, 1.5, 2.0),
    #                    help="Multi-scale inferences") 
    parser.add_argument("--weights_name", default=None, type=str, help="Model file name except directory. ex)resnet50_e150.pth")
    parser.add_argument("--cam_type", default='gradcam', type=str,
                        choices=['gradcam', 'gradcamplusplus', 'gradcam++', 'xgradcam', 'layercam',
                        'ablationcam', 'scorecam', 'eigencam', 'eigengradcam', 'fullgrad'])

    # Semi-supervsied
    parser.add_argument("--labeled_ratio", default=1., type=float)
    parser.add_argument("--use_unlabeled", action="store_true", help="Use unlabeled images after train_cam")
    
    # Step
    parser.add_argument("--finetune_skip", action="store_true")
    parser.add_argument("--gen_cam_skip", action="store_true")
    parser.add_argument("--eval_cam_skip", action="store_true")
    #parser.add_argument("--save_cam", action="store_true",
    #                    help="The flag which determines save cam before evaluation.")
    
    # args
    args = parser.parse_args()
    # voc12
    if args.dataset == 'voc12':
        args.voc_class = get_voc_class()
        args.voc_class_num = len(args.voc_class)

    # Run 
    # split random labeled labels (for Semi-supervised)
    if args.labeled_ratio < 1.:
        import split_label
        split_label.run(args)

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
