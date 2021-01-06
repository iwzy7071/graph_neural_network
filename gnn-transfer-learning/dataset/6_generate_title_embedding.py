import json
from sklearn.feature_extraction.text import HashingVectorizer

vectorizer = HashingVectorizer(n_features=4096, norm=None)


def generate_title_bad_of_words(name):
    path = f"./transfer/{name}.txt"
    print(path)
    paper_list = json.load(open(path))
    paper_titles = [paper['title'] for paper in paper_list]
    vec = vectorizer.fit_transform(paper_titles).toarray().tolist()
    json.dump(vec, open(f"./transfer/{name}.json", 'w'))


# generate_title_bad_of_words("paper_before_2002_clean")
# generate_title_bad_of_words("paper_betweeen_02and05_clean")
# generate_title_bad_of_words("paper_betweeen_05and08_clean")
# generate_title_bad_of_words("paper_betweeen_08and11_clean")
# generate_title_bad_of_words("paper_betweeen_11and14_clean")
# generate_title_bad_of_words("paper_betweeen_14and17_clean")
# generate_title_bad_of_words("paper_betweeen_17and20_clean")
