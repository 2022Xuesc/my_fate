import chardet

with open('coco_adj.pkl','r',encoding='Windows-1254') as f:
    print(chardet.detect(f.read()))