

# Paddle_PConv

Image Inpainting for Irregular Holes Using Partial Convolutions 论文复现

[English](./README.md) | 简体中文

   * [Paddle_PConv])
      * [一、简介](#一简介)
      * [二、复现精度](#二复现精度)
      * [三、数据集](#三数据集)
      * [四、环境依赖](#四环境依赖)
      * [五、快速开始](#五快速开始)
         * [step1:克隆](#克隆)
         * [step2:训练](#训练)
         * [step3:测试](#测试)
         * [使用预训练模型预测](#使用预训练模型预测)
      * [六、代码结构与详细说明](#六代码结构与详细说明)
         * [6.1 代码结构](#61-代码结构)
         * [6.2 参数说明](#62-参数说明)
         * [6.3 训练](#63-训练)
         * [6.4 评估和预测流程](#64-评估和预测流程)
      * [七、模型信息](#七模型信息)
## 一、简介
本项目基于paddlepaddle框架复现PartialConv，PartialConv是2018年ECCV的一篇关于图像修复的文章，主要方法是通过更新mask的策略来渐进式地对图像进行修复。此外作者提供了一个非规则mask的数据集，大约1.2w张mask图片。

**论文:**
- [1] Liu G, Reda F A, Shih K J, et al. Image inpainting for irregular holes  using partial convolutions[C]//Proceedings of the European Conference on Computer Vision (ECCV). 2018: 85-100.

**参考项目：**

- **官方源码**：https://github.com/NVIDIA/partialconv
- **非官方源码**：https://github.com/naoto0804/pytorch-inpainting-with-partial-conv
## 二、复现精度
本项目验收标准为Celeba-HQ数据集上人眼评估生成的图像，因为论文中没有足够的图片，所以又改成了在places2上的指标对比：下表是指标的对比，其中paper代表的是原论文中展示的指标效果，un-pytorch展示的是非官方的预训练模型的指标效果，our-paddle是我用paddle版本复现的效果。(PSNR和SSIM是越高越好,L1是越低越好)

![metrics](./images/metrics.png)

视觉效果：每张图的第一行是input，第二行是mask，第三行是非官方版的效果，第四行是paddle复现的效果，第五行是ground truth。

Dataset | maskratio:10-20 | maskratio:20-30 | maskratio:30-40 
:------:|:----------:|:------------------------:|:------------------------:
Places2|![result](./images/result2.jpg)|![result1](./images/result1.jpg)|![result2](./images/result.jpg)

CelebA-HQ的视觉效果：

这里我做测试的时候，随机从irregular mask数据集（1.2w张mask图片）中进行选择，对测试集进行修复：

<img src="./images/6.jpg" alt="6" style="zoom:5%;" />

## 三、数据集

[Celeba-HQ](http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html)+[Places2](http://places2.csail.mit.edu/)

### 数据集大小：

  - 训练集+验证集：分别用了2.9W 和 1W
  - 测试集：分别为1000张和100张
## 四、环境依赖
- 硬件：GPU、CPU

- 框架：
  - PaddlePaddle >= 2.0.0
## 五、快速开始
### 克隆
```bash
git clone https://github.com/maruru-hub/PartialConv-paddle.git
cd PartialConv-paddle
```
### 训练
```
python main.py 
```
### 测试
将模型的参数保存在```model\```中，然后改变pretrain_model的值，再运行以下命令，输出图片保存在```image\```目录中
```
python test.py
```
### 使用预训练模型预测

将需要测试的文件放在参数pretrain_model确定的目录下，修改option.py文件，运行下面指令
```
python main.py 
```
## 六、代码结构与详细说明

### 6.1 代码结构
因为本项目的验收是通过人眼观察图像，即user_study，因此评估脚本跟预测是同一个方式

```
├─datasets                                              # 数据集
├─logs                                                  # 训练日志
├─image                                                 # 训练时的可视化图像结果
├─checkpoints                                           # 模型参数文件
├─model                                                 # 模型
|  Decoder.py                                           # 解码器
|  Encoder.py                                           # 编码器
|  discriminator.py                                     # 判别器
|  PartialConv2d.py                                     # PartialConv方法实现
|  loss.py                                              # 损失函数
|  base_model.py                                        # 各种训练以及反向传播的初始化
│  README.md                                            # 英文readme
│  README_cn.md                                         # 中文readme
│  train.py                                             # 训练方法
│  test.py                                              # 测试方法
│  dataprocess.py                                       # 加载数据集
│  options.py                                           # 参数设置
```

### 6.2 参数说明

可以在 `option.py` 中设置训练与评估相关参数，具体如下：

|  参数   | 默认值  | 说明 | 其他 |
|  -------  |  ----  |  ----  |  ----  |
| batchSize | 4 | 加载数据批量 ||
| g_lr | 0.0002 | 生成器参数 |可以随着训练的进行递减|
| d_lr | 0.0002 | 判别器参数 ||
| --pretrain_model| None, 可选 | 预训练模型路径 ||
|### 6.3 训练||||
```bash
python main.py
```
#### 训练输出
执行训练开始后，训练内容会通过visualDL显示，通过下面代码查看日志文件

```text
visualdl --logdir ./logs/Celeba
visualdl --logdir ./logs/Place
```
### 6.4 评估和预测流程
我们的预训练模型如下：

这个是在celeba数据集下，lr=0.0002训练了60w个iterate的结果

链接：https://pan.baidu.com/s/1h6EQGLaHnrroZo91uTJXBw 
提取码：pdpd

这个是在Places2数据集下，lr=0.0002训练了60w个iterate的结果

链接：https://pan.baidu.com/s/1INLUYXRpJD_ywlPzH3IUGQ 
提取码：pdpd

## 七、模型信息

关于模型的其他信息，可以参考下表：

| 信息 | 说明 |
| --- | --- |
| 发布者 | 孙林 |
| 时间 | 2021.09 |
| 框架版本 | Paddle 2.0.2 |
| 应用场景 | 图像修复 |
| 支持硬件 | GPU、CPU |
# Log
```
visualdl --logdir ./logs/Celeba
visualdl --logdir ./logs/Place
```
# Results

视觉效果：每张图的第一行是input，第二行是mask，第三行是非官方版的效果，第四行是paddle复现的效果，第五行是ground truth。

| Dataset |         maskratio:10-20         |         maskratio:20-30          |         maskratio:30-40         |
| :-----: | :-----------------------------: | :------------------------------: | :-----------------------------: |
| Places2 | ![result](./images/result2.jpg) | ![result1](./images/result1.jpg) | ![result2](./images/result.jpg) |
