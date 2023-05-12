def classification_get_input_shape(dataset):
    if dataset == 'imagenet':
        return 1, 3, 224, 224
    if dataset == 'coco':
        return 1, 3, 448, 448
