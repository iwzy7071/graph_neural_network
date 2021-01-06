import json
from tqdm import tqdm

# Prepare target section
section = json.load(open('./transfer/label_count_top100.json'))
section = [sec[0] for sec in section]
pbar = tqdm(total=4894084, desc="read file")
allowed_section = [section[index] for index in [49, 46, 42, 43, 47, 48]]
paper_list = []
json_file = open('./transfer/dblp_v12.json', 'r', encoding='utf-8')

while True:
    line = json_file.readline()
    line = line.strip('\n').strip(',')
    if not line:
        break
    pbar.update(1)
    try:
        paper = json.loads(line)
        id, title, references, keywords = paper['id'], paper['title'], paper['references'], paper['fos']
        year = paper['year']
    except:
        continue
    keywords = [item['name'].lower() for item in keywords]
    if len(keywords) == 0:
        continue
    for label, section in enumerate(allowed_section):
        if section in keywords:
            paper_list.append({'id': id, 'title': title, 'year': year, 'reference': references, 'label': label})

# paper_index = [paper['id'] for paper in paper_list]
# for paper in reversed(paper_list):
#     for reference_paper in reversed(paper['reference']):
#         if reference_paper not in paper_index:
#             paper['reference'].remove(reference_paper)
#     if len(paper['reference']) == 0:
#         paper_list.remove(paper)

json.dump(paper_list, open('./transfer/dblp_v12_clean.txt', 'w', encoding='utf-8'))
