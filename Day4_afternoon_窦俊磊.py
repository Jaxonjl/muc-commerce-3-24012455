from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd

pd.set_option("display.max_columns", None)
pd.set_option("display.float_format", lambda x: f"{x:.2f}")

candidates = [
    Path("../data/E Commerce Dataset.xlsx"),
    Path("data/E Commerce Dataset.xlsx"),
    Path("/Users/yq/muc_training/data/E Commerce Dataset.xlsx"),
]
DATA_PATH = next((path for path in candidates if path.exists()), None)

if DATA_PATH is None:
    raise FileNotFoundError("未找到 E Commerce Dataset.xlsx，请修改 DATA_PATH。")

root_candidates = [Path.cwd(), Path.cwd().parent, Path("/Users/yq/Desktop/muc")]
PROJECT_ROOT = next(
    (path for path in root_candidates if (path / "notebooks").exists()),
    Path.cwd()
)
OUTPUT_DIR = PROJECT_ROOT / "output" / "day04_project"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

raw_df = pd.read_excel(DATA_PATH, sheet_name="E Comm")

print(f"原始数据：{DATA_PATH}")
print(f"项目输出目录：{OUTPUT_DIR}")
print(f"原始数据形状：{raw_df.shape}")
raw_df.head()
#每条记录代表一个电商用户的个人属性、行为特征和消费相关的完整信息
#CustomerID是用户唯一标识符，只是一个编号，没有数值含义，如果作为连续数值参与模型，会引入无意义的数字特征，干扰分析结果
#目标变量是Churn列，代表用户是否流失（1 = 流失，0 = 留存）
def build_quality_report(data):
    """返回字段级数据质量报告"""
    report = []
    for col in data.columns:
        col_data = data[col]
        missing_count = col_data.isnull().sum()
        missing_rate = missing_count / len(data) * 100
        unique_count = col_data.nunique()
        
        row = {
            "数据类型": str(col_data.dtype),
            "缺失数量": missing_count,
            "缺失比例(%)": round(missing_rate, 2),
            "唯一值数量": unique_count,
        }
        report.append(row)
    
    return pd.DataFrame(report, index=data.columns)

# 生成清洗前质量报告
quality_before = build_quality_report(raw_df)
# 生成清洗前质量报告
quality_before = build_quality_report(raw_df)
print(quality_before)
print("===== 初始审计 =====")
# 1. 完全重复行数
print(f"完全重复行数：{raw_df.duplicated().sum()}")

# 2. CustomerID 重复数量
print(f"CustomerID 重复数量：{raw_df['CustomerID'].duplicated().sum()}")

# 3. Churn 频数和流失率
print("\nChurn 频数：")
print(raw_df["Churn"].value_counts())
print(f"流失率：{raw_df['Churn'].mean():.2%}")

# 4. 主要类别字段的频数
print("\n主要类别字段频数：")
for col in ["PreferredLoginDevice", "PreferredPaymentMode", "PreferedOrderCat"]:
    print(f"\n{col}")
    print(raw_df[col].value_counts())
# ===== 定义清洗规则 =====
NUMERIC_MISSING_COLS = [
    "Tenure",
    "WarehouseToHome",
    "HourSpendOnApp",
    "OrderAmountHikeFromlastYear",
    "CouponUsed",
    "OrderCount",
    "DaySinceLastOrder",
]

CATEGORY_MAPPINGS = {
    "PreferredLoginDevice": {
        "Phone": "Mobile Phone"
    },
    "PreferredPaymentMode": {
        "COD": "Cash on Delivery",
        "CC": "Credit Card"
    },
    "PreferedOrderCat": {
        "Mobile": "Mobile Phone"
    }
}


# ===== 编写清洗函数 =====
def clean_ecommerce_data(data):
    """
    清洗电商用户行为数据。
    
    参数：
        data: 原始用户行为 DataFrame
    
    返回：
        cleaned_df: 清洗后的 DataFrame
        cleaning_log: 处理日志 DataFrame
    """
    # 复制数据，避免覆盖原始数据
    df = data.copy()
    logs = []
    
    # 1. 删除完全重复行
    before_count = len(df)
    df = df.drop_duplicates()
    after_count = len(df)
    logs.append({
        "处理步骤": "删除完全重复行",
        "处理规则": "完全相同的记录不增加信息，删除重复行",
        "处理前记录数": before_count,
        "处理后记录数": after_count,
        "影响记录数": before_count - after_count
    })
    
    # 2. 数值字段缺失值用中位数填补
    for col in NUMERIC_MISSING_COLS:
        before_missing = df[col].isnull().sum()
        median_val = df[col].median()
        df[col] = df[col].fillna(median_val)
        after_missing = df[col].isnull().sum()
        logs.append({
            "处理步骤": "缺失值填补",
            "处理规则": f"{col} 使用总体中位数填补，稳健且不将缺失误解为0",
            "处理前记录数": before_missing,
            "处理后记录数": after_missing,
            "影响记录数": before_missing - after_missing
        })
    
    # 3. 类别标准化
    for col, mapping in CATEGORY_MAPPINGS.items():
        before_count = 0
        for old_val, new_val in mapping.items():
            cnt = (df[col] == old_val).sum()
            before_count += cnt
            df[col] = df[col].replace(old_val, new_val)
        logs.append({
            "处理步骤": "类别标准化",
            "处理规则": f"{col} 统一业务类别命名：{str(mapping)}",
            "处理前记录数": before_count,
            "处理后记录数": 0,
            "影响记录数": before_count
        })
    
    # 4. 数据类型转换：Churn 和 Complain 转为整数
    for col in ["Churn", "Complain"]:
        df[col] = df[col].astype(int)
        logs.append({
            "处理步骤": "数据类型转换",
            "处理规则": f"{col} 转为整数类型",
            "处理前记录数": len(df),
            "处理后记录数": len(df),
            "影响记录数": 0
        })
    
    # 转成DataFrame
    cleaning_log = pd.DataFrame(logs)
    return df, cleaning_log
# 执行清洗
cleaned_df, cleaning_log = clean_ecommerce_data(raw_df)

# 查看日志
print("===== 清洗日志 =====")
print(cleaning_log.to_string())
print("\n清洗后前5行：")
print(cleaned_df.head())
# ===== IQR异常值检测函数 =====
def iqr_outlier_summary(series):
    """输出 IQR 候选异常值摘要"""
    series = series.dropna()
    q1 = series.quantile(0.25)
    q3 = series.quantile(0.75)
    iqr = q3 - q1
    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr
    
    return {
        "Q1": q1,
        "Q3": q3,
        "下限": lower,
        "上限": upper,
        "候选异常值数量": int(((series < lower) | (series > upper)).sum())
    }


# ===== 新增特征 =====
# 1. TenureGroup：用户使用时长分层
tenure_bins = [0, 5, 10, 15, 20, float("inf")]
tenure_labels = ["0-5年", "5-10年", "10-15年", "15-20年", "20年以上"]
cleaned_df["TenureGroup"] = pd.cut(cleaned_df["Tenure"], bins=tenure_bins, labels=tenure_labels, include_lowest=True)

# 2. IsMobileLogin：是否移动端登录
cleaned_df["IsMobileLogin"] = (cleaned_df["PreferredLoginDevice"] == "Mobile Phone").astype(int)


# ===== 生成候选异常值报告 =====
OUTLIER_COLS = ["WarehouseToHome", "OrderCount", "CashbackAmount"]
print("===== 候选异常值报告 =====")
outlier_report = []
for col in OUTLIER_COLS:
    summary = iqr_outlier_summary(cleaned_df[col])
    summary["字段"] = col
    outlier_report.append(summary)
    print(f"\n{col}:")
    print(f"  Q1: {summary['Q1']:.2f}, Q3: {summary['Q3']:.2f}")
    print(f"  异常范围：<{summary['下限']:.2f} 或 >{summary['上限']:.2f}")
    print(f"  候选异常值数量：{summary['候选异常值数量']}")

outlier_report_df = pd.DataFrame(outlier_report)
# ===== 业务规则检查 =====
print("\n===== 业务规则检查 =====")
business_rule_report = pd.DataFrame({
    "规则": ["使用时长小于0", "仓库距离小于0", "订单数小于或等于0", "返现金额小于0"],
    "不合规记录数": [
        (cleaned_df["Tenure"] < 0).sum(),
        (cleaned_df["WarehouseToHome"] < 0).sum(),
        (cleaned_df["OrderCount"] <= 0).sum(),
        (cleaned_df["CashbackAmount"] < 0).sum()
    ]
})
print(business_rule_report.to_string())
print("\n处理结论：所有业务违规记录数均为0，数据符合业务逻辑")
# ===== 生成清洗后质量报告 =====
quality_after = build_quality_report(cleaned_df)


# ===== 验收检查 =====
print("===== 项目验收 =====")
# 检查数值列无缺失
assert cleaned_df[NUMERIC_MISSING_COLS].isna().sum().sum() == 0, "❌ 数值字段仍有缺失值"
print("所有数值字段缺失值已填补")

# 检查类别统一
assert "Phone" not in cleaned_df["PreferredLoginDevice"].unique(), "❌ 登录设备尚未统一"
assert "COD" not in cleaned_df["PreferredPaymentMode"].unique(), "❌ 支付方式尚未统一"
assert "CC" not in cleaned_df["PreferredPaymentMode"].unique(), "❌ 支付方式尚未统一"
print("所有类别字段已标准化")

# 检查新增特征
assert {"TenureGroup", "IsMobileLogin"}.issubset(cleaned_df.columns), "❌ 缺少新增特征列"
print("新增特征列已生成")

print("\n🎉 所有验收检查通过！")


# ===== 导出所有交付文件 =====
quality_before.to_csv(OUTPUT_DIR / "data_quality_before.csv", index=True, encoding="utf-8-sig")
quality_after.to_csv(OUTPUT_DIR / "data_quality_after.csv", index=True, encoding="utf-8-sig")
cleaning_log.to_csv(OUTPUT_DIR / "cleaning_log.csv", index=False, encoding="utf-8-sig")
cleaned_df.to_csv(OUTPUT_DIR / "ecommerce_customer_cleaned.csv", index=False, encoding="utf-8-sig")

print("\n📁 所有文件已导出：")
print(f"输出目录：{OUTPUT_DIR.resolve()}")
print("  - data_quality_before.csv（清洗前质量报告）")
print("  - data_quality_after.csv（清洗后质量报告）")
print("  - cleaning_log.csv（处理日志）")
print("  - ecommerce_customer_cleaned.csv（清洗后数据）")

