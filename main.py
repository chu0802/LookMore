from src.train.trainer import main
from src.utils import Arg, argument_parser
from pathlib import Path

if __name__ == "__main__":
    args = argument_parser(
        Arg("-d", "--dataset_dir", type=Path, default=Path("/home/yuchuyu/project/lookwhere/output/subsample_maps/128000")),
        Arg("-s", "--seed", type=int, default=1102),
        Arg("-p", "--pretrained_params_path", type=str, default="models/lookwhere_dinov2.pt"),
        Arg("-n", "--num_epoches", type=int, default=10),
        Arg("-i", "--input_masked_ratio", type=float, default=0.2),
        Arg("-o", "--output_masked_ratio", type=float, default=0.0),
        Arg("-f", "--final_output_dir", type=Path, default=Path("/home/yuchuyu/project/lookwhere/models")),
    )
    
    args.final_output_dir.mkdir(parents=True, exist_ok=True)
    main(args)
