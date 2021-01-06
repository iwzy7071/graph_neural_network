import json
from tqdm import tqdm
import numpy as np

section = json.load(open('./transfer/label_count_top100.json'))
section = [sec[0] for sec in section]
section = section[:50]
matrix = np.zeros((50, 50), dtype=np.int)
json_file = open('./transfer/dblp_v12.json', 'r', encoding='utf-8')
pbar = tqdm(total=4894084, desc="read file")

while True:
    line = json_file.readline()
    line = line.strip('\n').strip(',')
    if not line:
        break
    pbar.update(1)
    try:
        paper = json.loads(line)
        keywords = paper['fos']
    except:
        continue
    keywords = [item['name'].lower() for item in keywords]
    for row in keywords:
        if row not in section:
            continue
        for col in keywords:
            if col not in section:
                continue
            matrix[section.index(row)][section.index(col)] += 1
print(matrix)
np.savetxt("dblp_v12_co_appear_matrix.txt", matrix)
