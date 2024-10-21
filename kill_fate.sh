ps -aux | grep /data/projects/fate/env/python36 | awk '{print $2}' | xargs kill
