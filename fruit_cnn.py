# 利用CNN实现水果分类

############################ 数据预处理 ############################
import os

name_dict = {"apple": 0, "banana": 1, "grape": 2, "orange": 3, "pear": 4}
# data_root_path = "data/fruits/"  # 数据集所在目录
data_root_path = "fruits_tiny/"
test_file_path = data_root_path + "test.txt"  # 测试集文件路径
train_file_path = data_root_path + "train.txt"  # 训练集文件路径
name_data_list = {}  # 记录每个类别有哪些图片  key:水果名称   value:存放图片路径列表


# 将图片路径存入name_data_list字典中
def save_train_test_file(path, name):
    if name not in name_data_list:  # 该类别水果不在字典中，新建一个字典并插入
        img_list = []
        img_list.append(path)
        name_data_list[name] = img_list  # 插入name-list键值对
    else:  # 该类别水果已经存在于字典中，直接添加到对应的列表
        name_data_list[name].append(path)


# 遍历每个子目录，拼接完整图片路径，并加入上述字典
dirs = os.listdir(data_root_path)
for d in dirs:
    full_path = data_root_path + d  # 拼接完整路径

    if os.path.isdir(full_path):  # 是一个子目录，读取其中的图片
        imgs = os.listdir(full_path)  # 列出子目录下所有的内容
        for img in imgs:
            save_train_test_file(full_path + "/" + img,  # 图片完整路径
                                 d)  # 以子目录名称作为类别名称
    else:  # 是一个文件，则不处理
        pass

# 遍历字典，划分训练集、测试集
## 清空训练集、测试集
with open(test_file_path, "w") as f:
    pass
with open(train_file_path, "w") as f:
    pass

## 遍历字典，划分训练集、测试集
for name, img_list in name_data_list.items():
    i = 0
    num = len(img_list)  # 获取每个类别样本数量
    print("%s: %d张" % (name, num))

    for img in img_list:
        if i % 10 == 0:  # 写入测试集
            with open(test_file_path, "a") as f:
                # 拼一行，格式： 图片路径  类别
                line = "%s\t%d\n" % (img, name_dict[name])
                f.write(line)
        else:  # 写入训练集
            with open(train_file_path, "a") as f:
                # 拼一行，格式： 图片路径  类别
                line = "%s\t%d\n" % (img, name_dict[name])
                f.write(line)

        i += 1  # 计数器加1

print("数据预处理完成.")

############################ 模型搭建、训练、保存 ############################
import paddle
import paddle.fluid as fluid
import numpy
import sys
import os
from multiprocessing import cpu_count
import time
import matplotlib.pyplot as plt


def train_mapper(sample):
    """
    根据传入的一行文本样本数据，读取相应的图像数据并返回
    :param sample: 元组，格式 (图片路径,类别)
    :return: 返回图像数据、类别
    """
    img, label = sample  # img为图像路径，label为所属的类别
    if not os.path.exists(img):
        print("图像不存在")

    # 读取图像数据
    img = paddle.dataset.image.load_image(img)
    # 对图像进行缩放，缩放到统一大小
    img = paddle.dataset.image.simple_transform(im=img,  # 原始图像数据
                                                resize_size=128,  # 图像缩放大小
                                                crop_size=128,  # 裁剪图像大小
                                                is_color=True,  # 彩色图像
                                                is_train=True)  # 训练模式，随机裁剪

    # 对图像数据进行归一化处理，将每个像素值转换到0~1之间
    img = img.astype("float32") / 255.0
    return img, label  # 返回图像数据(归一化处理后的)、类别


# 定义reader, 从训练集中读取样本
def train_r(train_list, buffered_size=1024):
    def reader():
        with open(train_list, "r") as f:
            lines = [line.strip() for line in f]  # 读取所有行，并去空格
            for line in lines:
                # 去除每行中的换行符，并按tab字符进行拆分
                img_path, lab = line.replace("\n", "").split("\t")
                yield img_path, int(lab)

    return paddle.reader.xmap_readers(train_mapper,  # 将reader读取到的数据进一步处理
                                      reader,  # 读取样本函数，读到数据送到train_mapper进一步处理
                                      cpu_count(),  # 线程数量(和逻辑CPU数量一致)
                                      buffered_size)  # 缓冲区大小

# 定义测试集读取器
def test_mapper(sample):
    img, label = sample

    img = paddle.dataset.image.load_image(img)
    img = paddle.dataset.image.simple_transform(im=img,
                                                resize_size=128,
                                                crop_size=128,
                                                is_color=True,
                                                is_train=False)
    img = img.astype("float32") / 255.0
    return img, label

def test_r(test_list, buffered_size=1024):
    def reader():
        with open(test_list, "r") as f:
            lines = [line.strip() for line in f]
            for line in lines:
                img_path, lab = line.split("\t")

                yield img_path, int(lab)

    return paddle.reader.xmap_readers(test_mapper,
                                      reader,
                                      cpu_count(),
                                      buffered_size)

# 定义reader
BATCH_SIZE = 16  # 批次大小
## 训练集reader
trainer_reader = train_r(train_list=train_file_path)  # 原始读取器
random_train_reader = paddle.reader.shuffle(reader=trainer_reader,
                                            buf_size=1300)  # 随机读取器
batch_train_reader = paddle.batch(random_train_reader,
                                  batch_size=BATCH_SIZE)  # 批量读取器
## 测试集reader
tester_reader = test_r(test_list=test_file_path) # 原始读取器
test_reader = paddle.batch(tester_reader, batch_size=BATCH_SIZE)# 批量读取器

# 变量
image = fluid.layers.data(name="image", shape=[3, 128, 128], dtype="float32")
label = fluid.layers.data(name="label", shape=[1], dtype="int64")


# 搭建CNN
# 结构：输入层 --> 卷积/激活/池化/dropout --> 卷积/激活/池化/dropout -->
#            卷积/激活/池化/dropout --> fc --> dropout --> fc(softmax)
def convolution_neural_network(image, type_size):
    """
    创建CNN
    :param image: 图像数据
    :param type_size: 分类数量
    :return: 一组分类概率(预测结果)
    """
    # 第一组 卷积/激活/池化/dropout
    conv_pool_1 = fluid.nets.simple_img_conv_pool(input=image,  # 输入数据，原始图像
                                                  filter_size=3,  # 卷积核大小3*3
                                                  num_filters=32,  # 卷积核数量
                                                  pool_size=2,  # 池化区域大小2*2
                                                  pool_stride=2,  # 池化步长值
                                                  act="relu")  # 激活函数
    drop = fluid.layers.dropout(x=conv_pool_1, dropout_prob=0.5)

    # 第二组 卷积/激活/池化/dropout
    conv_pool_2 = fluid.nets.simple_img_conv_pool(input=drop,  # 输入数据，上一个dropout输出
                                                  filter_size=3,  # 卷积核大小3*3
                                                  num_filters=64,  # 卷积核数量
                                                  pool_size=2,  # 池化区域大小2*2
                                                  pool_stride=2,  # 池化步长值
                                                  act="relu")  # 激活函数
    drop = fluid.layers.dropout(x=conv_pool_2, dropout_prob=0.5)

    # 第三组 卷积/激活/池化/dropout
    conv_pool_3 = fluid.nets.simple_img_conv_pool(input=drop,  # 输入数据，上一个dropout输出
                                                  filter_size=3,  # 卷积核大小3*3
                                                  num_filters=64,  # 卷积核数量
                                                  pool_size=2,  # 池化区域大小2*2
                                                  pool_stride=2,  # 池化步长值
                                                  act="relu")  # 激活函数
    drop = fluid.layers.dropout(x=conv_pool_3, dropout_prob=0.5)

    # fc
    fc = fluid.layers.fc(input=drop, size=512, act="relu")
    # dropout
    drop = fluid.layers.dropout(x=fc, dropout_prob=0.5)
    # fc
    predict = fluid.layers.fc(input=drop,
                              size=type_size,  # 输出值的个数(分类的数量)
                              act="softmax")
    return predict


# 调用函数，创建CNN
predict = convolution_neural_network(image=image, type_size=5)
# 损失函数
cost = fluid.layers.cross_entropy(input=predict,  # 预测结果
                                  label=label)  # 真实标签
avg_cost = fluid.layers.mean(cost)
# 准确率
accuracy = fluid.layers.accuracy(input=predict,  # 预测结果
                                 label=label)  # 真实标签
# 克隆(复制)一个program, 用于模型评估
test_program = fluid.default_main_program().clone(for_test=True)
# 优化器
optimizer = fluid.optimizer.Adam(learning_rate=0.001)
optimizer.minimize(avg_cost)
#  执行器
place = fluid.CUDAPlace(0)
exe = fluid.Executor(place)
exe.run(fluid.default_startup_program())
# feeder
feeder = fluid.DataFeeder(feed_list=[image, label],  # 指定要喂入的数据
                          place=place)

model_save_dir = "model/fruits/"  # 模型保存路径
costs = []  # 记录损失值
accs = []  # 记录准确率
batches = []  # 记录迭代次数
times = 0

# 开始训练
for pass_id in range(60):
    train_cost = 0  # 临时变量，记录每次训练的损失值
    for batch_id, data in enumerate(batch_train_reader()):  # 循环读取一批数据，执行训练
        times += 1
        train_cost, train_acc = exe.run(program=fluid.default_main_program(),
                                        feed=feeder.feed(data),  # 喂入参数
                                        fetch_list=[avg_cost, accuracy])  # 返回损失值、准确率
        if batch_id % 20 == 0:
            print("pass_id:%d, batch_id:%d, cost:%f, acc:%f" %
                  (pass_id, batch_id, train_cost[0], train_acc[0]))
            accs.append(train_acc[0])  # 记录准确率
            costs.append(train_cost[0])  # 记录损失值
            batches.append(times)  # 记录迭代次数

    # 模型评估
    test_accs = []
    test_costs = []

    for batch_id, data in enumerate(test_reader()):
        test_cost, test_acc = exe.run(program=test_program, # 执行用于测试的program
                                      feed=feeder.feed(data), # 喂入从测试集中读取的数据
                                      fetch_list=[avg_cost, accuracy]) # 获取预测损失值和准确率
        test_accs.append(test_acc[0])
        test_costs.append(test_cost[0])

    test_cost = (sum(test_costs) / len(test_costs)) # 求测试集下损失值的均值
    test_acc = (sum(test_accs) / len(test_accs))# 求测试集下准确率均值

    print("Test:%d, Cost:%f, Acc:%f" % (pass_id, test_cost, test_acc))


# 训练结束后，保存模型
if not os.path.exists(model_save_dir):
    os.makedirs(model_save_dir)  # 如果不存在则创建
fluid.io.save_inference_model(dirname=model_save_dir,  # 模型保存路径
                              feeded_var_names=["image"],  # 执行预测时需喂入的参数
                              target_vars=[predict],  # 预测结果从哪里取
                              executor=exe)  # 执行器
print("模型保存成功.")

# 训练过程可视化
plt.figure("training")
plt.title("training", fontsize=24)
plt.xlabel("iter", fontsize=14)
plt.ylabel("cost/acc", fontsize=14)
plt.plot(batches, costs, color="red", label="Training Cost")
plt.plot(batches, accs, color="green", label="Training Acc")
plt.legend()
plt.grid()
plt.savefig("train.png")
plt.show()


############################ 模型加载、预测 ############################
from PIL import Image

# 定义执行器
place = fluid.CPUPlace()
infer_exe = fluid.Executor(place)
model_save_dir = "model/fruits/"

# 加载图像数据
def load_img(path):
    # 读取待测试的图片数据
    img = paddle.dataset.image.load_and_transform(path, 128, 128, False).astype("float32")
    img = img / 255.0
    return img

infer_imgs = [] # 存放要预测图像数据
test_img = "apple_1.png" # 待预测图片路径
infer_imgs.append(load_img(test_img)) # 加载图像数据，并存入待预测列表
infer_imgs = numpy.array(infer_imgs) # 将列表转换为数组

# 加载模型
# 返回值含义：infer_program为预测时执行的program
#           feed_target_names预测时传入的参数
#           fetch_targets 预测结果从哪里获取
infer_program, feed_target_names, fetch_targets = \
    fluid.io.load_inference_model(model_save_dir, infer_exe)

# 执行预测
results = infer_exe.run(infer_program,
                        feed={feed_target_names[0]: infer_imgs}, # 喂入参数
                        fetch_list=fetch_targets)
print(results)

# 对预测结果进行转换
result = numpy.argmax(results[0]) # 取出预测结果，并将概率最大的索引值返回
for k, v in name_dict.items(): # 遍历字典，将数字转换为名称
    if result == v:
        print("预测结果:", k)

# 显示待预测图片
img = Image.open(test_img)
plt.imshow(img)
plt.show()