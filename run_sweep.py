import wandb

from config import sweep_config, wandb_config



def main():
    # 初始化wandb
    wandb.login()

    # 创建sweep
    sweep_id = wandb.sweep(
        sweep_config,
        project=wandb_config['project'],
        entity=wandb_config['entity']
    )
    
    print(f"Sweep ID: {sweep_id}")
    print("请在main.py中运行wandb.agent()来开始超参数搜索")

if __name__ == "__main__":
    main()