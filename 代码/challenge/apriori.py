#!/usr/bin/env python 
# -*- coding:utf-8 -*-
import pandas as pd
import numpy as np
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori

data = pd.read_csv("2.csv")
# te = TransactionEncoder()
# #进行 one-hot 编码
# te_ary = te.fit(data).transform(data)
# df = pd.DataFrame(te_ary, columns=te.columns_)
# 利用 Apriori 找出频繁项集
freq = apriori(data, min_support=0.05)
freq.to_csv("apriori.csv", index=False)

