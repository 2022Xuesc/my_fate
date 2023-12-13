import os.path
import csv


class MyWriter(object):
    def __init__(self, dir_name,stats_name='stats'):
        super(MyWriter, self).__init__()
        self.stats_dir = os.path.join(dir_name,stats_name)
        if not os.path.exists(self.stats_dir):
            os.makedirs(self.stats_dir)

    def get(self, file_name, buf_size=1, header=''):
        # 根据文件路径和buffer_size创建文件对象
        file = open(os.path.join(self.stats_dir, file_name), 'w', buffering=buf_size)
        writer = csv.writer(file)
        # 写入表头信息，如果有的话
        if len(header) != 0:
            writer.writerow(header)
        return writer
