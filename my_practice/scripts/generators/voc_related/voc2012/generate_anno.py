import json
import os


def generate_anno(data, images_dir, phase='train'):
    image_id_path = os.path.join(data, f'{phase}_full_image_id.json')
    image_id = json.load(open(image_id_path, 'r'))
    files = os.listdir(images_dir)
    anno_list = []
    for filename in files:
        if not filename.endswith('.jpg'):
            continue
        anno_list.append(image_id[filename])
    target_file_path = os.path.join(images_dir, 'anno.json')
    json.dump(anno_list, open(target_file_path, 'w'))


client_nums = 10

# image_dir = '/data/projects/dataset/voc2007/clustered_voc_expanded'
# category_dir = '/data/projects/fate/my_practice/dataset/voc_expanded'

# 客户端
image_dir = 'images'
category_dir = '/home/klaus125/research/fate/my_practice/dataset/voc_expanded'

for i in range(client_nums):
    client_id = i + 1
    print('generate anno for ',client_id)
    generate_anno(category_dir, os.path.join(image_dir, f'client{client_id}/train'), phase='train')
    generate_anno(category_dir, os.path.join(image_dir, f'client{client_id}/val'), phase='val')
    
    # generate_anno(category_dir,image_dir,phase='train')
