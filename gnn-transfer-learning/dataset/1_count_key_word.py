from tqdm import tqdm
import json

json_file = open('./transfer/dblp_v12.json', 'r', encoding='utf-8')
pbar = tqdm(total=4894084, desc="read file")
paper_list = []
# allowed_sections = ['database', 'data mining', 'artificial intelligent', 'computer vision', 'information security',
#                     'high performance computing']
label_count = {}

while True:
    line = json_file.readline()
    line = line.strip('\n').strip(',')
    if not line:
        break
    pbar.update(1)
    try:
        paper = json.loads(line)
        id, title, references, keywords, year = paper['id'], paper['title'], paper['references'], paper['fos'], paper[
            'year']
    except:
        continue
    keywords = [item['name'].lower() for item in keywords]
    if len(keywords) == 0:
        continue
    for keyword in keywords:
        label_count.setdefault(keyword, 0)
        label_count[keyword] += 1
    # for label, section in enumerate(allowed_sections):
    #     if section in keywords:
    #         paper_list.append({'id': id, 'title': title, 'year': year, 'reference': references, 'label': label})
    #         label_count.setdefault(label, 0)
    #         label_count[label] += 1
    #         break

json.dump(label_count, open('./transfer/label_count.json', 'w'))
# paper_index = [paper['id'] for paper in paper_list]
# for paper in tqdm(reversed(paper_list), desc='cleaning'):
#     for reference_paper in reversed(paper['reference']):
#         if reference_paper not in paper_index:
#             paper['reference'].remove(reference_paper)
#     if len(paper['reference']) == 0:
#         paper_list.remove(paper)
#
# file = open('./transfer/dblp_v12_clean.json', 'w', encoding='utf-8')
# file.writelines(paper_list)
#
