import itertools
import pandas as pd


def load_data(path):  # 文件所在的路径
    content = []  # 用来存放处理后的内容
    with open(path) as f:
        for line in f:
            line = line.strip("\n")
            content.append(line.split(","))
    return content


def buildC1(dataset):
    item1 = set(itertools.chain(*dataset))
    return [frozenset([i]) for i in item1]


def ck_to_lk(dataset, ck, min_support):
    support = {}  # 定义项集-频数字典，用来存储每个项集(key)与其对应的频数(value)
    for row in dataset:
        for item in ck:
            if item.issubset(row):  # 判断项集是否在记录（行）中出现
                support[item] = support.get(item, 0) + 1
    total = len(dataset)
    return {k: v / total for k, v in support.items() if v / total >= min_support}


def lk_to_ck(lk_list):
    ck = set()  # 保存所有组合之后的候选k+1项集
    lk_size = len(lk_list)
    if lk_size > 1:  # 如果频率k项集的数量小于1，则不可能通过组合生成k+1项集，直接返回空set即可
        k = len(lk_list[0])  # 获取频繁k项集的k值
        for i, j in itertools.combinations(range(lk_size), 2):
            t = lk_list[i] | lk_list[j]  # 将对应位置的两个频繁k项集进行组合，生成一个新的项集
            if len(t) == k + 1:  # 如果组合之后的项集是k+1项集，则为候选k+1项集，加入结果到set中
                ck.add(t)
    return ck


def get_L_all(dataset, min_support):
    c1 = buildC1(dataset)
    L1 = ck_to_lk(dataset, c1, min_support)

    L_all = L1  # 定义字典，保存所有的频繁k项集
    Lk = L1
    while len(Lk) > 1:  # 当频繁项集中的元素（键值对）大于1时，才有可能组合生成候选k+1项集
        lk_key_list = list(Lk.keys())
        ck = lk_to_ck(lk_key_list)  # 由频繁k项集生成候选k+1项集
        Lk = ck_to_lk(dataset, ck, min_support)  # 由候选k+1项集生成频繁k+1项集
        if len(Lk) > 0:  # 如果频繁k+1项集字典不为空，则将所有频繁k+1项集加入到L_all字典中
            L_all.update(Lk)
        else:
            break  # 否则，频繁k+1项集为空，退出循环
    return L_all


def rules_from_item(item):
    left = []  # 定义规则左侧的列表
    for i in range(1, len(item)):
        left.extend(itertools.combinations(item, 1))
    return [(frozenset(le), frozenset(item.difference(le))) for le in left]


def rules_from_L_all(L_all, min_confidence):
    rules = []  # 保存所有候选的关联规则
    for Lk in L_all:
        # 如果频繁项集的元素个数为1，则无法生成关联规则，不予考虑
        if len(Lk) > 1:
            rules.extend(rules_from_item(Lk))
    result = []
    for left, right in rules:
        support = L_all[left | right]
        confidence = support / L_all[left]
        lift = confidence >= min_confidence
        if confidence > min_confidence:
            result.append({"左侧": left, "右侧": right, "支持度": support, "置信度": confidence, "提升度": lift})
    return result


def apriori(dataset, min_support, min_confidence):
    L_all = get_L_all(dataset, min_support)
    rules = rules_from_L_all(L_all, min_confidence)
    return rules


def change(item):
    li = list(item)
    for i in range(len(li)):
        li[i] = index_to_str[li[i]]
    return li


if __name__ == '__main__':
    dataset = load_data("Market_Basket_Optimisation.txt")
    print(len(dataset))
    for i in range(10):
        print(i + 1, dataset[i], sep=",")
    items = set(itertools.chain(*dataset))  # 二维列表扁平化
    str_to_index = {}  # 用来保存字符串到编码的映射
    index_to_str = {}  # 用来保存编码到字符串的映射

    for index, item in enumerate(items):
        str_to_index[item] = index
        index_to_str[index] = item

    # 输出结果
    for item in list(str_to_index.items())[:5]:
        print(item)
    for item in list(index_to_str.items())[:5]:
        print(item)

    for i in range(len(dataset)):
        for j in range(len(dataset[i])):
            dataset[i][j] = str_to_index[dataset[i][j]]
    for i in range(10):  # 输出结果
        print(i + 1, dataset[i], sep="->")

    rules = apriori(dataset, 0.05, 0.3)

    df = pd.DataFrame(rules)
    df.reindex(["左侧", "右侧", '支持度', '置信度', '提升度'], axis=1)
    df["左侧"] = df["左侧"].apply(change)
    df["右侧"] = df["右侧"].apply(change)
    print(df)
