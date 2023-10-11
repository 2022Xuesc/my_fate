import os


def generate_configs(dir_paths):
    if not isinstance(dir_paths, list):
        dir_paths = [dir_paths]
    for dir_path in dir_paths:
        config_path = os.path.join(dir_path, 'config.yaml')
        if not os.path.exists(config_path):
            file = open(config_path, 'w')
            file.close()


client_nums = 10
image_dir = "/data/projects/voc2007/clustered_voc"
for i in range(client_nums):
    client_id = i + 1
    generate_configs(os.path.join(image_dir, f'client{client_id}/val'))
    generate_configs(os.path.join(image_dir, f'client{client_id}/train'))
