import json
import os
import shutil


def generate_anno(data, images_dir, phase='train'):
    image_id_path = os.path.join(data, f'{phase}_image_id.json')
    image_id = json.load(open(image_id_path, 'r'))
    files = os.listdir(images_dir)
    anno_list = []
    for filename in files:
        if not filename.endswith('.jpg'):
            continue
        key = int(filename.split('.')[0].split('_')[-1])
        anno_list.append(image_id[str(key)])
    target_file_path = os.path.join(images_dir, 'anno.json')
    json.dump(anno_list, open(target_file_path, 'w'))


client_ids = range(1, 11)
target_dir = '/data/projects/dataset/coco2017/clustered_dataset/global_val'
for client_id in client_ids:
    val_dir = f'/data/projects/dataset/coco2017/clustered_dataset/client{client_id}/val'
    # 遍历val_dir下的所有图片，将其拷贝到target_dir中
    file_names = os.listdir(val_dir)
    for file_name in file_names:
        if file_name.endswith('.jpg'):
            shutil.copy(os.path.join(val_dir, file_name), target_dir)

# Todo: 为其生成.json文件

category_dir = '/data/projects/fate/my_practice/dataset/coco2017'
generate_anno(category_dir, target_dir, phase='val')
