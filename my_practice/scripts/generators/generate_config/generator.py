import json
import os.path

data_dir = '/data/projects/clustered_dataset'
namespace = 'coco-clients10'
type = 'non-iid'

client_num = 10
phases = ['train', 'val']
dir_path = f"../../multi-label/server/bind_data/{type}"
if not os.path.exists(dir_path):
    os.makedirs(dir_path)

bind_name = os.path.join(dir_path, "bind_all_data.sh")
f = open(bind_name, 'w')
f.write("#!/bin/bash\n")

for i in range(1, client_num + 1):
    for phase in phases:
        filename = f'bind_client{i}_{phase}_path.json'
        f.write(f"flow table bind --drop -c {filename}\n")
        file_path = os.path.join(dir_path, filename)
        print(file_path)
        obj = {
            "engine": "PATH",
            "namespace": namespace,
            "name": f"client{i}_{phase}",
            "address": {
                "path": f"{data_dir}/client{i}/{phase}"
            }
        }
        json_str = json.dumps(obj, indent=4)
        with open(file_path, 'w') as json_file:
            json_file.write(json_str)
