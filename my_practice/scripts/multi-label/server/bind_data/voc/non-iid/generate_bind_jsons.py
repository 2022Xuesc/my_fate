import json

num_clients = 10
base_path = "/data/projects/voc2007/clustered_voc"
for i in range(1, num_clients + 1):
    for phase in {'train', 'val'}:
        json_name = f'bind_client{i}_{phase}_path.json'
        content = dict()
        content["engine"] = "PATH"
        content["namespace"] = "voc2007-clients10"
        content["name"] = f'client{i}_{phase}'
        address = dict()
        address["path"] = f'{base_path}/client{i}/{phase}'
        content["address"] = address
        print(content)
        json_file = open(json_name,'w')
        json.dump(content,json_file)
