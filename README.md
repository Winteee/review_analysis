# review_analysis
### 利用Keras进行评论的情感极性分析
- 此为Winteeena的本科毕业论文选题，通过Keras提供的框架进行中文情感判断训练（也是我用python跑通的第一个程序 (^_^)）
- 数据来源：大众点评-上海美食商户的30000条评论（review30000.csv）

### 实验结果
- 针对单条评论的正负性预测准确率达93%

# 实验过程
## 预先安装的
- Anaconda（Python 3.5）
- Tensorflow
## 文档结构
- ReadMe
- Review-Sentiment-Analysis
- - final_review_analysis.ipynb：在笔记本上运行的结果
- - final_review_analysis.py：包含所有代码的PY文件
- - review30000.csv：实验用的评论数据（来自大众点评）
- - traindata.csv：经过文本预处理后的中间文件，用于进行数据训练
- - predict_data.csv：模型验证完成后，输入200条新评论，机器预测的结果

# 参考文档
1. Pandas中文文档：http://wiki.jikexueyuan.com/project/start-learning-python/311.html
2. Keras中文文档：http://keras-cn.readthedocs.io/en/latest/

