from pprint import pprint
from paddlenlp import Taskflow

#执行不同类别的任务，控制台显示调用的包也不同，仔细观察，可供改进。
# relation:
#schema = [{'Person': ['Company', 'Position']}]#改person为Man也可行，：代表关系，对比多任务抽取可知
#'In 1997, Steve was excited to become the CEO of Apple.' 
 
# Define the schema for opinion extraction 
#schema = [  {'Aspect': ['Opinion', 'Sentiment classification [negative, positive]']}  ] #英文的评价维度及其对应的观点词和情感倾向，可以发现和relation十分类似
# ********
#schema = [  {'AAAAAAAAAAAAAAAA': ['Opinion', 'Sentiment classification [negative, positive]']}  ] #if we change the name  X'Aspect'X ,it doesn't work. It seems that 'Aspect' works like 'Person'.
#"The teacher is very nice."

#my test:
#'In 1997, Steve was excited to become the CEO of Apple.' 
#schema = 'Steve is the [Position] of Apple'

#情感分类
#"I am sorry but this is the worst film I have ever seen in my life."
#schema =["Sentiment classification [negative, positive]"] #中括号可有可无

#3.6 跨任务抽取（代改进原中文实例）
#schema = ['法院', {'原告': '委托代理人'}, {'被告': '委托代理人'}]

schema = ['Court',{'Plaintiff':'Entrusted agent'},{'Defendant':'Entrusted agent'}]
#Civil Judgment of Beijing Haidian District People's Court (1999) Jianchuzi No. 052. Plaintiff: Zhang SAN. Entrusted agent Li Si, lawyer of Beijing A Law Firm. Defendant: B Company, legal representative Wang Wu, general manager of Development Company. Entrusted agent Zhao Liu, lawyer of Beijing C Law Firm.

# 模型选择
#uie-base (默认)	12-layers, 768-hidden, 12-heads	中文
#uie-base-en	12-layers, 768-hidden, 12-heads	英文
#uie-medical-base	12-layers, 768-hidden, 12-heads	中文
#uie-m-large	24-layers, 1024-hidden, 16-heads	中、英文  支持中英文混合抽取
#uie-m-base	12-layers, 768-hidden, 12-heads	中、英文  支持中英文混合抽取

#Anything Else
#from paddlenlp import Taskflow

#ie = Taskflow('information_extraction',
#schema="", #定义任务抽取目标
#schema_lang="zh",#设置schema的语言，默认为zh, 可选有zh和en。因为中英schema的构造有所不同，因此需要指定schema的语言。该参数只对uie-m-base和uie-m-large模型有效。
#batch_size=1, #批处理大小，请结合机器情况进行调整，默认为1
#model='uie-base',
#position_prob=0.5, #模型对于span的起始位置/终止位置的结果概率在0~1之间，返回结果去掉小于这个阈值的结果，默认为0.5，span的最终概率输出为起始位置概率和终止位置概率的乘积。
#precision='fp32') #选择模型精度，默认为fp32，可选有fp16和fp32。fp16推理速度更快。如果选择fp16，请先确保机器正确安装NVIDIA相关驱动和基础软件，确保CUDA>=11.2，cuDNN>=8.1.1，初次使用需按照提示安装相关依赖。

ie_en = Taskflow('information_extraction', schema=schema, model='uie-base-en')
ie_en.set_schema(schema)
pprint(ie_en("Civil Judgment of Beijing Haidian District People's Court (1999) Jianchuzi No. 052.  Defendant: B Company, legal representative Wang Wu, general manager of Development Company. Entrusted agent Zhao Liu, lawyer of Beijing C Law Firm."))
print(len("Civil Judgment of Beijing Haidian District People's Court (1999) Jianchuzi No. 052. Plaintiff: Zhang SAN. Entrusted agent Li Si, lawyer of Beijing A Law Firm. Defendant: B Company, legal representative Wang Wu, general manager of Development Company. Entrusted agent Zhao Liu, lawyer of Beijing C Law Firm."))