import os
import shutil

jobs_dir = '../../../fateflow/jobs'
logs_dir = '../../../fateflow/logs'

job_id =  "202303211528593528620"

arbiter_path = "arbiter/10000/multi_label_0"
guest_path = "guest/9999/multi_label_0"
host_path = "host/9998/multi_label_0"


def del_session_stops(job_id):
    # 组合路径
    paths = [arbiter_path, guest_path, host_path]
    job_dir = os.path.join(jobs_dir, job_id)
    for path in paths:
        deldir = os.path.join(job_dir, path)
        # 删除dir路径下的session_stop文件夹
        shutil.rmtree(os.path.join(deldir, 'session_stop'))


def extract_params(job_id):
    log_dir = os.path.join(logs_dir, job_id)
    filename = 'fate_flow_schedule.log'
    file_path = os.path.join(log_dir, filename)
    fp = open(file_path, 'r')
    pattern = "--component_name multi"
    param = '--job_id'
    for line in fp:
        if pattern in line:
            print(line[line.find(param):])


del_session_stops(job_id)

extract_params(job_id)