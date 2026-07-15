from pathlib import Path
import pandas as pd

DATA_DIR = Path('data')
CSV_PATH = DATA_DIR / '淘宝全品类全国数据.csv'

print('当前工作目录：', Path.cwd())
print('数据文件存在：', CSV_PATH.exists())

df = pd.read_csv(CSV_PATH)
"""
print(df.shape)



print('数据规模：', df.shape)
print('字段名：', df.columns.tolist())
print(df.head(10))
#把整个表格的前10行打印出来让我们看一下基本的情况
#然后现实有10行15列 也就是我前面让他打印出来的10行15列
print(df.info())
#先显示我的数据类型 是data frame 一共有两万五千行 展示了十五个列的名称并统计出每一列有多少有效数据 这些数据有什么类型
# 这里说明除了价格是浮点数类型 其他全都是字符串类型 也就是14列的字符串 一列的浮点数


print(df.dtypes)
#具体展示每一列究竟是什么类型

missing_count = df.isna().sum().sort_values(ascending=False)
print(missing_count)
#isna用来把空的变成True 不是空的变成False 然后每一列求和算出每一列有多少缺失数据
# 最后sort_values利用这个将求和的值ascending=False利用这个降序去排序

# 缺失率（百分比）
missing_rate = (df.isna().mean() * 100).round(1).sort_values(ascending=False)
print(missing_rate)
#这里我们将缺失率很大的列放弃数值统计比如版型和面料 没有什么意义 缺失率很低或者为0比如价格和面料可以进行数值统计

# 一列：Series
price_series = df['商品价格']
print(type(price_series))

# 多列：DataFrame
product_view = df[['商品id', '一级品类', '商品价格', '省份', '商品销量']]
print(type(product_view))
print(product_view.head(10))
#默认是展示选取的这几列的五行 我改成10行去印证一下

print(df.loc[0:4, ['一级品类', '商品价格', '省份']])
print(df.iloc[0:5, 0:4])
#这里loc是把第0到第5行 标签为选区的三个给取出来 iloc是只看绝对的第几行第几列 这里就是从第0行开始取前五行 从第0行开始 取前4列


# 单条件
guangdong = df[df['省份'] == '广东']
print(guangdong)
#筛选出广东发货的有多少件商品 我打印出来看了一下

# 多条件：每个条件都要加括号，使用 & 连接
condition = (df['省份'] == '广东') & (df['商品价格'] >= 1000)

selected = df.loc[condition, ['商品id', '一级品类', '二级品类', '商品价格', '省份', '商品销量']]
selected = selected.sort_values(by='商品价格', ascending=False)
print(selected.head(10))
#先在df里选择广东加商品价格大于等于一万的商品 再选择选取的六列 然后按照商品价格进行降序排序


# 或条件
zhejiang_or_jiangsu = df[(df['省份'] == '浙江') | (df['省份'] == '江苏')]
print('浙江或江苏商品数：', zhejiang_or_jiangsu.shape[0])
#shape[0]如果只写shape的话就是给一个元组 （多少行多少列）但是我们加上【0】这个索引就变成只要多少行就行


# 商品价格的描述性统计
print(df['商品价格'].describe().round(2))

# 一级品类商品数
print(df['一级品类'].value_counts())

# 一级品类汇总
category_summary = (
    df.groupby('一级品类')
      .agg(商品数=('商品id', 'size'),
           平均价格=('商品价格', 'mean'),
           中位价格=('商品价格', 'median'))
      .sort_values('平均价格', ascending=False)
      .round(2)
)
print(category_summary)
#按照一级品类取分类 然后计算每个品类有多少个，也就是商品数是多少，然后商品价格的平均值， 商品价格的中位数 
"""


provinces = ['山东', '北京']
subset = df[df['省份'].isin(provinces)]

province_summary = (
    subset.groupby('省份')
          .agg(商品数=('商品id', 'size'),
               平均价格=('商品价格', 'mean'),
               中位价格=('商品价格', 'median'))
          .round(2)
)
print(province_summary)

for province in provinces:
    top_category = (subset.loc[subset['省份'] == province, '一级品类']
                         .value_counts()
                         .head(1))
    print('\n', province, '最常见一级品类：')
    print(top_category)
