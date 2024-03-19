import random

def generate_random_disease_probabilities():
    diseases = [
        "斑点落叶病",
        "褐斑病",
        "健康",
        "锈病",
        "炭疽叶枯病",
        "轮斑病",
        "苹果霉心病",
        "腐烂病",
        "灰斑病",
        "花叶病",
        "黑星病",
        "白粉病",
        "圆斑病",
        "黑点红病",
        "白粉锈病"
    ]

    # 随机选择最大概率对应的疾病
    max_prob_disease = random.choice(diseases)

    # 生成最大概率
    max_probability = random.uniform(0.4, 1.0)
    probabilities = [max_probability]

    # 生成剩余概率
    remaining_probability = 1.0 - max_probability
    for _ in range(len(diseases) - 2):
        probability = random.uniform(0, remaining_probability)
        probabilities.append(probability)
        remaining_probability -= probability

    # 最后一个概率为剩余的概率
    probabilities.append(remaining_probability)

    # 将最大概率移到随机选择的位置
    health_pro = random.uniform(0.4, 1.0)
    if health_pro > 0.6:
        max_prob_index = 2
        max_prob_value = probabilities.pop(0)
        probabilities.insert(max_prob_index, max_prob_value)
    else:

        max_prob_index = diseases.index(max_prob_disease)
        max_prob_value = probabilities.pop(0)
        probabilities.insert(max_prob_index, max_prob_value)

    # 随机打乱概率列表
        random.shuffle(probabilities)

    # 构建字典
    disease_probabilities = dict(zip(diseases, probabilities))

    return disease_probabilities



def generate_random_pest_probabilities():
    pests = [
        "绵蚜",
        "越冬虫卵",
        "绿盲蝽",
        "金龟子",
        "红蜘蛛",
        "苹果食心虫",
        "卷叶虫",
        "蚜虫",
        "金蚊细蛾",
        "食心虫"
    ]

    # 随机选择最大概率对应的虫害
    max_prob_pest = random.choice(pests)

    # 生成最大概率
    max_probability = random.uniform(0.4, 0.9)
    probabilities = [max_probability]

    # 生成剩余概率
    remaining_probability = 1.0 - max_probability
    for _ in range(len(pests) - 2):
        probability = random.uniform(0, remaining_probability)
        probabilities.append(probability)
        remaining_probability -= probability

    # 最后一个概率为剩余的概率
    probabilities.append(remaining_probability)

    # 将最大概率移到随机选择的位置
    max_prob_index = pests.index(max_prob_pest)
    max_prob_value = probabilities.pop(0)
    probabilities.insert(max_prob_index, max_prob_value)

    # 随机打乱概率列表
    random.shuffle(probabilities)

    # 构建字典
    pest_probabilities = dict(zip(pests, probabilities))

    return pest_probabilities

# 示例
# random_probabilities = generate_random_pest_probabilities()
# print(random_probabilities)
def generate_random_pest_quantities():
    pests = [
        "绵蚜",
        "越冬虫卵",
        "绿盲蝽",
        "金龟子",
        "红蜘蛛",
        "苹果食心虫",
        "卷叶虫",
        "蚜虫",
        "金蚊细蛾",
        "食心虫"
    ]

    # 随机选择虫害的数量
    num_pests = random.randint(3, 5)

    # 随机选择1到3种虫害
    selected_pests = random.sample(pests, random.randint(1, 3))

    # 构建字典，表示各种虫害的数量
    pest_quantities = {pest: 0 for pest in pests}
    for pest in selected_pests:
        pest_quantities[pest] = random.randint(1, 10)

    return pest_quantities
