import json

json_path = 'val_image2labels.json'
image_id2labels = json.load(open(json_path, 'r'))

# 标签i和j有强相关性
s1 = set()
# 标签i到j有强相关性，而j到i没有强相关性
s2 = set()
# 标签j到i有强相关性质，而i到j没有强相关性
s3 = set()
# 标签i和j之间没有强相关性
s4 = set()


def add_list_to_set(s, images):
    for image in images:
        s.add(image)


num_labels = 80
# 遍历每一个标签对
cnt = 0
for i in range(num_labels):
    for j in range(i + 1, num_labels):
        # 建立3个列表
        images_ij = []  # i和j同时出现
        images_i = []  # 仅包含i
        images_j = []  # 仅包含j
        # 遍历每张图片，根据含有的标签将图像分到上述3个列表中
        for image_id in image_id2labels:
            image_info = image_id2labels[image_id]
            filename = image_info['file_name']
            label = set(image_info['labels'])
            # 对label进行分析，加入到以上三个列表之一
            if i in label and j in label:
                images_ij.append(filename)
            elif i in label:
                images_i.append(filename)
            elif j in label:
                images_j.append(filename)
        # 判断完成后，将ij中[0,1/2]的图片放到s1中，[1/2,3/4]放到s2，剩下放到s3中
        half_point_ij = len(images_ij) // 2
        three_quarter_point = len(images_ij) * 3 // 4
        add_list_to_set(s1, images_ij[:half_point_ij])
        add_list_to_set(s1, images_ij[half_point_ij:three_quarter_point])
        add_list_to_set(s1, images_ij[three_quarter_point:])

        # 将i中的一半放到s3,剩下一半放到s4中
        half_point_i = len(images_i) // 2
        add_list_to_set(s3, images_i[:half_point_i])
        add_list_to_set(s4, images_i[half_point_i:])
        # 将j中的一般放到s2，剩下一般放到s4中
        half_point_j = len(images_j) // 2
        add_list_to_set(s3, images_j[:half_point_j])
        add_list_to_set(s4, images_j[half_point_j:])
# 最后根据s1,s2,s3,s4划分数据集
print('Done')
