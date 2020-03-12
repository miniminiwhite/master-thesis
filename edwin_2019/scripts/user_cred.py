import os
import sys
import json
import math
import threading
from collections import defaultdict
import pandas as pd

from unionFind import UnionFind


THRESHOLD = 0.6
if len(sys.argv) > 1:
    THRESHOLD = float(sys.argv[1])


def date_intersect(start_a, end_a, start_b, end_b):
    if start_a > end_b or start_b > end_a:
        return None
    return sorted([start_a, end_a, start_b, end_b])[1:3]


# Multi-thread speed up.
class MyThread(threading.Thread):
    threads = []
    lock = threading.Lock()
    sem = threading.Semaphore(200)
    user_cmt_data = None
    user_lst_data = None
    uf = None

    def __init__(self, cur_idx):
        threading.Thread.__init__(self)
        self.idx = cur_idx

    @staticmethod
    def run_tasks():
        for thread in MyThread.threads:
            MyThread.sem.acquire()
            thread.start()

        for thread in MyThread.threads:
            thread.join()

    @classmethod
    def set_uf(cls, uf):
        cls.uf = uf

    @classmethod
    def set_lst_data(cls, lst_data):
        cls.user_lst_data = lst_data

    @classmethod
    def set_cmt_data(cls, cmt_data):
        cls.user_cmt_data = cmt_data

    def run(self):
        cur_usr_a = MyThread.user_lst_data[self.idx]
        print(f"Thread {self.idx} start.")
        for j in range(self.idx + 1, len(MyThread.user_lst_data)):
            # print(f"{j}", end="\t")
            cur_usr_b = MyThread.user_lst_data[j]
            # Calculate similarity
            d_usr_a = MyThread.user_cmt_data[cur_usr_a]
            d_usr_b = MyThread.user_cmt_data[cur_usr_b]
            time_range = date_intersect(d_usr_a["s"], d_usr_a["e"], d_usr_b["s"], d_usr_b["e"])
            if not time_range:
                # pass this pair if there's no overlap between the dates
                continue
            duration = (time_range[1] - time_range[0]).days + 1
            accu = 0
            duration_none_empty = 0
            for k in range(duration):
                cur_date = time_range[0] + pd.Timedelta(k, unit="d")
                cmt_pkgs_a = d_usr_a["data"].get(cur_date, {})
                cmt_pkgs_b = d_usr_b["data"].get(cur_date, {})
                # Calculate Jaccard's coefficient
                frac_a = 0
                frac_b = sum(cmt_pkgs_a.values()) + sum(cmt_pkgs_b.values())
                if not frac_b:
                    continue
                for pkg in cmt_pkgs_a.keys():
                    if pkg in cmt_pkgs_b.keys():
                        frac_a += min(cmt_pkgs_a[pkg], cmt_pkgs_b[pkg])
                accu += frac_a / (frac_b - frac_a)
                duration_none_empty += 1
            similarity = accu / duration_none_empty
            if similarity > THRESHOLD:
                MyThread.lock.acquire()
                MyThread.uf.union(self.idx, j)
                MyThread.lock.release()
        print(f"Thread {self.idx} finished.")
        MyThread.sem.release()


f = open(os.path.join("data", "test.dat"), "r", encoding="utf-8")
data = sorted([line[:-1].split("\t") for line in f], key=lambda x: x[3])
f.close()

print("Initializing...")
user_cmt_data = {}
for line in data:
    line[4] = pd.to_datetime(line[4])
    if line[3] not in user_cmt_data:
        user_cmt_data[line[3]] = {
            "s": None,
            "e": None,
            "data": {},
            "cnt": 0,
        }
    if not user_cmt_data[line[3]]["s"] or line[4] < user_cmt_data[line[3]]["s"]:
        user_cmt_data[line[3]]["s"] = line[4]
    if not user_cmt_data[line[3]]["e"] or line[4] > user_cmt_data[line[3]]["e"]:
        user_cmt_data[line[3]]["e"] = line[4]
    if line[4] not in user_cmt_data[line[3]]["data"]:
        user_cmt_data[line[3]]["data"][line[4]] = defaultdict(int)
    user_cmt_data[line[3]]["data"][line[4]][line[1]] += 1
    user_cmt_data[line[3]]["cnt"] += 1

del_list = []
for k, v in user_cmt_data.items():
    if v["cnt"] < 2:
        del_list.append(k)
print(f"Delete {len(del_list)} user(s) from {len(user_cmt_data)} users.")
for k in del_list:
    del user_cmt_data[k]

user_lst = sorted(list(user_cmt_data.keys()))
uf = UnionFind(len(user_lst))

MyThread.set_cmt_data(user_cmt_data)
MyThread.set_lst_data(user_lst)
MyThread.set_uf(uf)

for i in range(len(user_lst) - 1):
    MyThread.threads.append(MyThread(i))

MyThread.run_tasks()

result = defaultdict(list)
for idx, res in enumerate(uf.result()):
    result[res].append(user_lst[idx])

# print("Writing result.")
if not os.path.exists("res"):
    os.mkdir("res")
    
f = open(os.path.join("res", f"usr_cred_weight_res_{THRESHOLD}.json"), "w", encoding="utf-8")
json.dump({group: {
    "Group weight": math.sqrt(len(mem)),
    "Mem weight": math.sqrt(len(mem))/len(mem)
} for group, mem in result.items()}, f, indent=2)
f.close()

f = open(os.path.join("res", f"usr_cred_grouping_res_{THRESHOLD}.json"), "w", encoding="utf-8")
json.dump({group: mem for group, mem in result.items()}, f, indent=2, ensure_ascii=False)
f.close()

# print("Done.")
