from monai.bundle import ConfigParser

config = ConfigParser()
config.read_config("./training_configs/training_config_template.yaml")

training_config = config.get_parsed_content("ambf_train_config")
workspace = config.get_parsed_content("workspace", eval_expr=True)
data_dir = config.get_parsed_content("data_dir", eval_expr=True)

print("check paths...")
print(f"workspace: {workspace.exists()}")
print(f"data_dir: {training_config['data_dir'].exists()}")
print(f"train_dir: { [p.exists() for p in training_config['train_dir_list']] }")
print(f"val_dir: { [p.exists() for p in training_config['val_dir_list']] }")


print(f"{config.get_parsed_content('test')}")
x = 0
