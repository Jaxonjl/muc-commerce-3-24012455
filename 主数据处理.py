from pathlib import Path
import pandas as pd

DATA_DIR = Path('data')
EXCEL_PATH = DATA_DIR / 'E Commerce Dataset.xlsx'
print('当前工作目录：', Path.cwd())
print('数据文件存在：', EXCEL_PATH.exists())

df = pd.read_excel(EXCEL_PATH,sheet_name="E Comm")

print('数据规模：', df.shape)
print('字段名：', df.columns.tolist())
print(df.head(10))
print(df.info())
print(df.dtypes)
#进行前期检查
missing_count = df.isna().sum().sort_values(ascending=False)
print(missing_count)
missing_rate = (df.isna().mean() * 100).round(1).sort_values(ascending=False)
print(missing_rate)
#统计缺失率
"""
num_cols = df.select_dtypes(include=['int64','float64']).columns
for col in num_cols:
    median_val = df[col].median()
    df[col] = df[col].fillna(median_val)
#然后检查有无没有填充上的
print("各列缺失值数量：")
print(df.isnull().sum())
#对数字类型数据的缺失处进行中位数填充
"""

"""
# 要填充的数字列列表
numeric_missing_cols = [
    "Tenure",
    "WarehouseToHome",
    "HourSpendOnApp",
    "OrderAmountHikeFromlastYear",
    "CouponUsed",
    "OrderCount",
    "DaySinceLastOrder",
]
#循环填充每列的中位数
for col in numeric_missing_cols:
    # 1. 计算当前列的中位数（自动跳过其中的 NaN）
    median_val = df[col].median()
    # 2. 使用该中位数填充当前列的缺失值，并覆盖原数据
    df[col] = df[col].fillna(median_val)

#输出上述字段剩余的缺失值数量
print("验证剩余缺失值数量：")
print(df[numeric_missing_cols].isna().sum())


#对数字类型数据的缺失处进行中位数填充
cat_cols = df.select_dtypes(include=['object', 'string']).columns
for col in cat_cols:
    mode_val = df[col].mode()[0]
    df[col] = df[col].fillna(mode_val)
#这里是用来众数填充的 比如一些字符串类型数据或者01数据

# 先定义需要处理的分类字段列表
category_cols = [
    "PreferredLoginDevice",
    "PreferredPaymentMode",
    "PreferedOrderCat"
]
"""
"""
# 1. 自动把全表的列分为两拨：数字列 和 非数字列
numeric_cols = df.select_dtypes(include=['number']).columns
non_numeric_cols = df.select_dtypes(exclude=['number']).columns

# 2. 自动化流水线：数字列全部用【中位数】填充
for col in numeric_cols:
    median_val = df[col].median()
    df[col] = df[col].fillna(median_val)

# 3. 自动化流水线：非数字列（文本/object/string）全部用【'未知'】填充
# 这样既消灭了 NaN，又绝对不会报错，确保逻辑安全
for col in non_numeric_cols:
    df[col] = df[col].fillna('未知')

# 4. 验证全表是否还有剩余的缺失值
print("目前整张表各列的缺失值总数：")
print(df.isna().sum())
"""

# ========== 1. 完成类别标准化替换 ==========
# 登录设备：Phone → Mobile Phone
df["PreferredLoginDevice"] = df["PreferredLoginDevice"].replace("Phone", "Mobile Phone")

# 支付方式：COD→Cash on Delivery、CC→Credit Card
df["PreferredPaymentMode"] = df["PreferredPaymentMode"].replace({"COD": "Cash on Delivery", "CC": "Credit Card"})

# 订单品类：Mobile → Mobile Phone
df["PreferedOrderCat"] = df["PreferedOrderCat"].replace("Mobile", "Mobile Phone")

# ========== 2. 重新检查并输出处理后的频数 ==========
for col in category_cols:
    print(f"\n{col}")
    print(df[col].value_counts())

# 1. 统计整行完全重复的行数（首次出现保留，后续重复行标记为True）
duplicate_rows = df.duplicated().sum()

# 2. 统计CustomerID的重复记录数量（同一个ID除首次外的重复条数）
duplicate_customer_ids = df['CustomerID'].duplicated().sum()

# 打印结果
print("完全重复行数：", duplicate_rows)
print("CustomerID重复数量：", duplicate_customer_ids)


def iqr_outlier_summary(series):
#返回数值字段的 IQR 候选异常值摘要
    series = series.dropna()
    q1 = series.quantile(0.25)
    q3 = series.quantile(0.75)
    iqr = q3 - q1
    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr

    return pd.Series({
        "Q1": q1,
        "Q3": q3,
        "下限": lower,
        "上限": upper,
        "候选异常值数量": ((series < lower) | (series > upper)).sum()
    })
result1 = iqr_outlier_summary(df["WarehouseToHome"])
print("==== WarehouseToHome ====")
print(result1)

result2 = iqr_outlier_summary(df["OrderCount"])
print("\n==== OrderCount ====")
print(result2)

result3 = iqr_outlier_summary(df["CashbackAmount"])
print("\n==== CashbackAmount ====")
print(result3)

#完成业务规则检查
rules = {
    "使用时长小于0": (df["Tenure"] < 0).sum(),
    "仓库距离小于0": (df["WarehouseToHome"] < 0).sum(),
    "订单数小于或等于0": (df["OrderCount"] <= 0).sum(),
    "返现金额小于0": (df["CashbackAmount"] < 0).sum(),
}

result = pd.Series(rules)
print("===== 业务违规记录统计 =====")
print(result)
#清洗结果验收
#清洗结果验收
numeric_missing_cols = ["WarehouseToHome", "OrderCount", "CashbackAmount"]
#检查数值字段缺失值
assert df[numeric_missing_cols].isna().sum().sum() == 0, "数值字段仍有缺失值"
#检查同义类别是否清理完毕
assert "Phone" not in df["PreferredLoginDevice"].unique(), "登录设备尚未统一"
assert "COD" not in df["PreferredPaymentMode"].unique(), "支付方式尚未统一"
assert "CC" not in df["PreferredPaymentMode"].unique(), "支付方式尚未统一"

print("数据清洗验收通过。")

OUTPUT_PATH = Path("data/ecommerce_customer_cleaned.csv")
OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

df.to_csv(OUTPUT_PATH, index=False, encoding="utf-8-sig")
print(f"已导出：{OUTPUT_PATH.resolve()}")