import json
import pandas
from tqdm import tqdm

all_paper_list = json.load(open('./transfer/dblp_v12_clean.txt', encoding='utf-8'))
paper_02, paper_02_05, paper_05_08, paper_08_11, paper_11_14, paper_14_17, paper_17_20 = [], [], [], [], [], [], []
for paper in all_paper_list:
    year = paper['year']
    if year < 2002:
        paper_02.append(paper)
    elif 2002 < year <= 2005:
        paper_02_05.append(paper)
    elif 2005 < year <= 2008:
        paper_05_08.append(paper)
    elif 2008 < year <= 2011:
        paper_08_11.append(paper)
    elif 2011 < year <= 2014:
        paper_11_14.append(paper)
    elif 2014 < year <= 2017:
        paper_14_17.append(paper)
    elif 2017 < year <= 2020:
        paper_17_20.append(paper)


def clean_dataset(paper_list, name):
    paper_index = [paper['id'] for paper in paper_list]
    edge_count = 0
    pbar = tqdm(total=len(paper_list), desc=name)
    for paper in reversed(paper_list):
        pbar.update(1)
        for reference_index in reversed(paper['reference']):
            if reference_index not in paper_index:
                paper['reference'].remove(reference_index)
        if len(paper['reference']) == 0:
            paper_list.remove(paper)
        else:
            edge_count += len(paper['reference'])
    print(f"{name}:{len(paper_list)}:{edge_count}")
    json.dump(paper_list, open(f"./transfer/{name}_clean.txt", 'w'))


clean_dataset(paper_02, "paper_before_2002")
clean_dataset(paper_02_05, "paper_betweeen_02and05")
clean_dataset(paper_05_08, "paper_betweeen_05and08")
clean_dataset(paper_08_11, "paper_betweeen_08and11")
clean_dataset(paper_11_14, "paper_betweeen_11and14")
clean_dataset(paper_14_17, "paper_betweeen_14and17")
clean_dataset(paper_17_20, "paper_betweeen_17and20")
