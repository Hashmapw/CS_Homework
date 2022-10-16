from efficient_apriori import apriori


def load_data(path):  # 文件所在的路径
    content = []  # 用来存放处理后的内容
    with open(path, encoding="UTF-8") as f:
        for line in f:
            line = line.strip("\n")
            content.append(line.split(","))
    return content


if __name__ == '__main__':
    dataset = load_data("data.txt")
    print(len(dataset))
    for i in range(10):
        print(i + 1, dataset[i], sep=",")
    itemSets, rules = apriori(dataset, min_support=0.05, min_confidence=0.3)
    print(itemSets)
    print(rules)