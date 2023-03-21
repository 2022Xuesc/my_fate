import json
import os.path

client_num = 8
phases = ['train', 'val']

dir_path = "../../multi-label/server/bind_data"
bind_name = os.path.join(dir_path, "bind_all_data.sh")
f = open(bind_name, 'w')
f.write("#!/bin/bash\n")

for i in range(1, client_num + 1):
    for phase in phases:
        filename = f'bind_client{i}_{phase}_path.json'
        f.write(f"flow table bind --drop -c {filename}\n")
        file_path = os.path.join(dir_path, filename)
        obj = {
            "engine": "PATH",
            "namespace": "ms-coco-0321",
            "name": f"client{i}_{phase}",
            "address": {
                "path": f"/data/projects/my_dataset/client{i}/{phase}"
            }
        }
        json_str = json.dumps(obj, indent=4)
        with open(file_path, 'w') as json_file:
            json_file.write(json_str)
