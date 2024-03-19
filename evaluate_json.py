import json



j = './submit/baseline.json'
# 读取JSON文件
with open(j, 'r') as file:
    data = json.load(file)

# 初始化空列表用于存储结果
result_x = []
result_y = []

# 遍历每个对象
for item in data:
    # 提取第一个值的前两位数字，并删除第一个0
    x = str(int(item["image_id"][:2]))
    # 提取第二个值的数组
    y = item["disease_class"]

    # 将结果添加到列表
    result_x.append(x)
    result_y.append(y)

# 打印结果
#print("x:", result_x)
#print("y:", result_y)

result = []
for i in range(len(result_x)):
    result_x[i] = str(result_x[i])
    result_y[i] = str(result_y[i])
    if result_x[i] == result_y[i]:
        result.append(True)
    else:
        result.append(False)

print("ACC:", end=' ')
print(100 *  (1.0*result.count(True)) / len(result_x)) 