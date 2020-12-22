import os
import os.path as osp
import shutil

path = osp.join(osp.dirname(osp.realpath(__file__)), 'bad_case')
dir_name = ['GAT_cora', 'GCN_cora', 'SAGE_cora', 'SGC_cora']
dir_path = [osp.join(path, name) for name in dir_name]

pic_names = [set([name for name in os.listdir(path)]) for path in dir_path]

union_set = pic_names[0]
print(union_set)
for oth_set in pic_names[1:]:
    print(len(oth_set))
    union_set = union_set & oth_set
print(len(union_set))

for name in list(union_set):
    pic_path = osp.join(dir_path[0], name)
    shutil.copyfile(pic_path, osp.join('/home/wzy/graph_neural_network/bad_case/public', name))
