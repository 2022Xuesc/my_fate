# jobs/202303230755355571510/arbiter/1/multi_label_0/202303230755355571510_multi_label_0/0/task_executor/2ef98ae4c95011edabbc3cecefb44c14/stats
# jobs/202303230755355571510/arbiter/0/multi_label_0/202303230755355571510_multi_label_0/0/task_executor
#
# jobs/202303230755355571510/guest/2/multi_label_0/202303230755355571510_multi_label_0/0/task_executor/2f8fdc9cc95011edae143cecefb44c14/stats
#
#
#
# jobs/202303230755355571510/host/3/multi_label_0/202303230755355571510_multi_label_0/0/task_executor/2f0a225ac95011eda7043cecefb44c14/stats
# jobs/202303230755355571510/host/4/multi_label_0/202303230755355571510_multi_label_0/0/task_executor/2f1c3dd2c95011edae3a3cecefb44c14/stats


# 到task_executor后直接进入目录即可
import os
import shutil

dir_id = "202401020908108100370"
job_id = "202401020908108100370"
module_name = 'gcn_0'
target_dir = 'IJCNN/salgl_agg'
client_num = 10


# 当前目录为xxx/task_executor，进入到[子目录/stats]中，拷贝其中的avgloss.csv,train.csv,valid.csv文件
def mv_files(dir_path, target_path):
    files = os.listdir(dir_path)
    files_dir = os.path.join(dir_path, f'{files[0]}/stats')
    for filename in os.listdir(files_dir):
        file_path = os.path.join(files_dir, filename)
        if os.path.isfile(file_path):
            shutil.copy(file_path, target_path)


def mv_stats(role, role_ids, target_dir):
    if not isinstance(role_ids, list):
        role_ids = [role_ids]
    for role_id in role_ids:
        dir_path = f'/data/projects/fate/fateflow/jobs/{dir_id}/{role}/{role_id}/{module_name}/{job_id}_{module_name}/0/task_executor'
        # 目标路径
        target_path = f'{target_dir}/{role}/{role_id}'
        if not os.path.exists(target_path):
            os.makedirs(target_path)
        mv_files(dir_path, target_path)


mv_stats('arbiter', 999, target_dir)
mv_stats('guest', 10, target_dir)
mv_stats('host', list(range(1, client_num)), target_dir)










