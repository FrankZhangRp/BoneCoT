import os
import argparse
from omegaconf import OmegaConf
import pathlib
import trainer
algorithm_names = sorted(name for name in trainer.__dict__ if 'Trainer' in name and callable(trainer.__dict__[name]))


def get_args():
    # 创建 ArgumentParser 对象
    parser = argparse.ArgumentParser(description="DinoV2 finetune")
    
    # 添加命令行参数
    parser.add_argument("--config-file", type=str, required=True, help="Model configuration file", default="/data/dataserver01/zhangruipeng/code/BoneFM/our_finetune/configs/default_configs.yaml")
    parser.add_argument("--output-dir", default="", type=str, help="Output directory to write results and logs")
    parser.add_argument("--log-interval", type=int, help="Log interval", default=10)
    parser.add_argument("--log_display", action="store_true", help="Whether to display log")
    

    # 解析命令行参数
    args = parser.parse_args()

    # 加载默认配置文件和用户指定的配置文件
    default_config_path = pathlib.Path(__file__).parent.resolve() / "default_configs.yaml"
    default_cfg = OmegaConf.load(default_config_path)
    
    # 如果提供了配置文件路径，加载用户配置文件
    user_cfg = OmegaConf.load(args.config_file)

    # 合并配置文件：使用用户配置覆盖默认配置，但保留默认配置中的其他条目
    merged_cfg = OmegaConf.merge(default_cfg, user_cfg)

    # 将合并后的配置应用到 args 中
    for key, value in merged_cfg.items():
        if not hasattr(args, key):
            setattr(args, key, value)
        else:
            # 优先使用命令行参数，但如果命令行中没有提供，则使用配置文件中的值
            if getattr(args, key) is None:
                setattr(args, key, value)

    # 更新 output_dir 为绝对路径
    args.output_dir = os.path.abspath(args.output_dir)

    # 如果 output-dir 不存在，则创建目录
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    # 将最终的 args 写入到 output_dir 下的 config.yaml 文件中，并按照键名排序
    args_dict = vars(args)  # 将 args 转换为字典
    sorted_args_dict = dict(sorted(args_dict.items()))  # 按键名排序

    # 保存到 YAML 文件中
    config_yaml_path = os.path.join(args.output_dir, "config.yaml")
    with open(config_yaml_path, "w") as f:
        OmegaConf.save(config=OmegaConf.create(sorted_args_dict), f=f)

    # 返回合并后的 args
    return args


# 使用示例
if __name__ == "__main__":
    # 获取命令行参数和配置
    args = get_args()
    print(args.batch_size)