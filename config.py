import argparse

arg_lists = []
parser = argparse.ArgumentParser()

def add_argument_group(name):
    arg = parser.add_argument_group(name)
    arg_lists.append(arg)
    return arg

# Dataset
data_arg = add_argument_group('Dataset')
data_arg.add_argument('--dataset', type=str, default='vimeo90K_septuplet')
data_arg.add_argument('--data_root', type=str, default='/home/zhihao/DATA-M2/video_interpolation/vimeo_septuplet/')

# data_arg.add_argument('--dataset', type=str, default='Davis')
# data_arg.add_argument('--data_root', type=str, default='/home/zhihao/DATA-M2/video_interpolation/Davis/')

# data_arg.add_argument('--dataset', type=str, default='ucf101')
# data_arg.add_argument('--data_root', type=str, default='/home/zhihao/DATA-M2/video_interpolation/UCF/')

# Model
model_arg = add_argument_group('Model')
model_choices = ["VFIT_S", "VFIT_B"]
model_arg.add_argument('--model', choices=model_choices, type=str, default="VFIT_B")
model_arg.add_argument('--nbr_frame' , type=int , default=4)
model_arg.add_argument('--joinType' , choices=["concat" , "add" , "none"], default="concat")

# Training / test parameters
learn_arg = add_argument_group('Learning')
learn_arg.add_argument('--loss', type=str, default='1*L1')
learn_arg.add_argument('--lr', type=float, default=2e-4)
learn_arg.add_argument('--beta1', type=float, default=0.9)
learn_arg.add_argument('--beta2', type=float, default=0.999)
learn_arg.add_argument('--batch_size', type=int, default=4)
learn_arg.add_argument('--test_batch_size', type=int, default=12)
learn_arg.add_argument('--start_epoch', type=int, default=0)
learn_arg.add_argument('--max_epoch', type=int, default=100)
learn_arg.add_argument('--resume', action='store_true')
learn_arg.add_argument('--resume_exp', type=str, default=None)
learn_arg.add_argument('--checkpoint_dir', type=str ,default=".")
learn_arg.add_argument("--load_from"  ,type=str , default='checkpoints/VFIT_B/model_best.pth')
learn_arg.add_argument("--pretrained" , type=str,
                        help="Load from a pretrained model.")

# Misc
misc_arg = add_argument_group('Misc')
misc_arg.add_argument('--exp_name', type=str, default='exp')
misc_arg.add_argument('--log_iter', type=int, default=100)
misc_arg.add_argument('--num_gpu', type=int, default=1)
misc_arg.add_argument('--random_seed', type=int, default=103)
misc_arg.add_argument('--num_workers', type=int, default=1)
misc_arg.add_argument('--val_freq', type=int, default=1)

def get_args():
    """Parses all of the arguments above
    """
    args, unparsed = parser.parse_known_args()
    if args.num_gpu > 0:
        setattr(args, 'cuda', True)
    else:
        setattr(args, 'cuda', False)
    if len(unparsed) > 1:
        print("Unparsed args: {}".format(unparsed))
    return args, unparsed
