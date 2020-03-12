import os
import sys
import json
import threading
from collections import defaultdict

import jieba

from textSimilarity import TextSimilarity as Ts
from unionFind import UnionFind as Uf


LEASE_COMMENT_CNT = 50
WORD_DROP_THRESHOLD = 2
T3 = 0.8
if len(sys.argv) > 3:
    LEASE_COMMENT_CNT = int(sys.argv[1])
    WORD_DROP_THRESHOLD = int(sys.argv[2])
    T3 = float(sys.argv[3])


def cluster_cmts(documents, threshold):
    uf = Uf(len(documents))
    for i in range(len(documents) - 1):
        for j in range(i + 1, len(documents)):
            if jaccard_coefficient(documents[i], documents[j]) >= threshold:
                uf.union(i, j)

    res = defaultdict(list)
    for idx, cluster in enumerate(uf.result()):
        res[cluster].append(documents[idx])

    return res


def jaccard_coefficient(a, b):
    """
    :param a: list of items
    :param b: list of items.
    :return: the jaccard coefficient
    """
    sa = set(a)
    sb = set(b)
    return len(sa & sb) / len(sa | sb)


def main(all=["test.dat"], pos="test.dat"):
    raw_data = []
    for file in all:
        f = open(os.path.join("data", file), "r", encoding="utf-8")
        raw_data += [line[:-1].split("\t") for line in f]
        f.close()

    f = open(os.path.join("data", pos), "r", encoding="utf-8")
    pos_data = [line[:-1].split("\t") for line in f]
    f.close()

    # Count package all original comment number
    pkg_ori_cmt_cnt = defaultdict(int)
    for entry in raw_data:
        pkg_ori_cmt_cnt[entry[1]] += 1

    cut_comments = defaultdict(list)
    word_cnt = defaultdict(int)
    documents = defaultdict(set)
    screened_data = []

    # Cut sentence and count word frequency
    # Only for pkgs with cmts more than threshold
    for idx, entry in enumerate(pos_data):
        if pkg_ori_cmt_cnt[entry[1]] > LEASE_COMMENT_CNT:
            cut_res = Ts.cut_sentence(entry[2])
            cut_comments[cut_res].append(idx)
            for word in cut_res:
                word_cnt[word] += 1

    # Remove word of frequency less than WORD_DROP_THRESHOLD
    # Delete comment of length 0 after removing low-frequency words
    for k, v in cut_comments.items():
        word_list = tuple(word for word in k if word_cnt[word] > WORD_DROP_THRESHOLD)
        if not word_list:
            continue
        for idx in v:
            screened_data.append(pos_data[idx])
            documents[pos_data[idx][1]].add(word_list)

    for i in documents:
        assert len(i)

    print(f"Selected {len(screened_data)} valid comments from {len(pos_data)} comments.")

    pkg_cmt = defaultdict(list)
    for entry in screened_data:
        pkg_cmt[entry[1]].append(entry)

    clusters = {}
    duplicate_rate = {}
    for pkg in pkg_cmt:
        if len(pkg_cmt[pkg]) < 50:
            continue
        clusters[pkg] = cluster_cmts(list(documents[pkg]), T3)
        duplicate_rate[pkg] = 1 - len(clusters[pkg]) / len(pkg_cmt[pkg])

    # Output
    if not os.path.exists("res"):
        os.mkdir("res")

    folder = f"cmt_simi_{LEASE_COMMENT_CNT}_{WORD_DROP_THRESHOLD}_{T3}"
    if not os.path.exists(os.path.join("res", folder)):
        os.mkdir(os.path.join("res", folder))

    f = open(os.path.join("res", folder, f"cluster_res_{T3}.json"), "w", encoding="utf-8")
    json.dump(clusters, f, indent=2, ensure_ascii=False)
    f.close()

    f = open(os.path.join("res", folder, f"comment_in_pkg_{WORD_DROP_THRESHOLD}.json"), "w", encoding="utf-8")
    json.dump(pkg_cmt, f, indent=2, ensure_ascii=False)
    f.close()

    f = open(os.path.join("res", folder, f"duplicate_rate_{WORD_DROP_THRESHOLD}_{T3}.json"), "w", encoding="utf-8")
    json.dump(duplicate_rate, f, indent=2, ensure_ascii=False)
    f.close()

    res = sorted([(k, v) for k, v in duplicate_rate.items()], key=lambda x: (-x[1], x[0]))
    for idx, items in enumerate(res):
        if items[0] in [
            "com.jhwl.fjxa",
            "com.arsenal.FunWeather",
            "com.sscwap",
        ]:
            print(idx, items[0], items[1])


if __name__ == "__main__":
    data_list = [f"360_{i+1}_star_comments.dat" for i in range(5)]
    main(data_list, "360_5_star_comments.dat")
    # main()
