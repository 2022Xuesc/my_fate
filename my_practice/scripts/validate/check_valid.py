import os

dir_id = "202410201704402101840"
job_id = "202410201704402101840"
module_name = 'gcn_0'
target_dir = 'AAAI/voc2007/fixed_connect_prob_standard_gcn'
client_num = 10


# 当前目录为xxx/task_executor，进入到[子目录/stats]中，拷贝其中的avgloss.csv,train.csv,valid.csv文件
def check_valid(dir_path):
    files = os.listdir(dir_path)
    train_file = os.path.join(dir_path, f'{files[0]}/stats/train.csv')
    with open(train_file, 'r') as file:
        cnt = 0
        for line in file:
            parts = line.strip().split(',')
            if parts[-1].strip() == 'nan':
                return False
            if cnt == 4:
                return True
            cnt += 1
    return True


def check(role, role_ids):
    if not isinstance(role_ids, list):
        role_ids = [role_ids]
    for role_id in role_ids:
        dir_path = f'/data/projects/fate/fateflow/jobs/{dir_id}/{role}/{role_id}/{module_name}/{job_id}_{module_name}/0/task_executor'
        if not check_valid(dir_path):
            print(f'{role}-{role_id} is not valid')


check('guest', [10])
check('host', list(range(1, client_num)))
