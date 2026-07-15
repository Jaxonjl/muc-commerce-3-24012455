from pathlib import Path
import pandas as pd
import numpy as np
try:
    from IPython.display import display
except ImportError:
    def display(obj):
        print(obj)

pd.set_option("display.max_columns", 50)
pd.set_option("display.float_format", lambda x: f"{x:,.2f}")

def find_workspace_root(start=None):
    """从当前目录向上寻找项目根目录。"""
    start = Path.cwd() if start is None else Path(start)
    for candidate in [start, *start.parents]:
        if (candidate / "output" / "day04_project" / "ecommerce_customer_cleaned.csv").exists():
            return candidate
    raise FileNotFoundError("未找到项目根目录，请确认第4天清洗结果已生成。")

ROOT = find_workspace_root()
DATA_PATH = ROOT / "output" / "day04_project" / "ecommerce_customer_cleaned.csv"
OUTPUT_DIR = ROOT / "output" / "day05_analysis"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

print("项目根目录：", ROOT)
print("输入数据：", DATA_PATH)
print("输出目录：", OUTPUT_DIR)

df = pd.read_csv(DATA_PATH)

core_cols = [
    "CustomerID", "Churn", "Tenure", "TenureGroup", "OrderCount",
    "CouponUsed", "CashbackAmount", "DaySinceLastOrder", "Complain",
    "PreferedOrderCat", "PreferredPaymentMode"
]

validation = pd.Series({
    "行数": len(df),
    "列数": df.shape[1],
    "CustomerID重复数": int(df["CustomerID"].duplicated().sum()),
    "核心字段缺失数": int(df[core_cols].isna().sum().sum()),
    "Churn取值": sorted(df["Churn"].unique().tolist()),
}, name="验收结果")

display(validation.to_frame())
display(df.head())

assert df.shape == (5630, 22), "数据形状与第4天交付物不一致"
assert df["CustomerID"].is_unique, "CustomerID存在重复"
assert df[core_cols].notna().all().all(), "核心字段仍有缺失"
assert set(df["Churn"].unique()) == {0, 1}, "Churn应只包含0和1"
print("数据验收通过：一行代表一名用户。")

metric_dictionary = pd.DataFrame([
    ["用户数", "CustomerID", "nunique", "独立用户数量"],
    ["流失人数", "Churn", "sum", "Churn=1的用户数量"],
    ["流失率", "Churn", "mean", "当前分组中流失用户占比"],
    ["平均订单数", "OrderCount", "mean", "用户级平均订单次数"],
    ["平均优惠券数", "CouponUsed", "mean", "用户级平均优惠券使用次数"],
    ["平均返现", "CashbackAmount", "mean", "返现金额，不等于消费金额"],
    ["平均距上次下单天数", "DaySinceLastOrder", "mean", "越大通常表示近期活跃度越低"],
], columns=["指标名称", "字段", "聚合方式", "解释边界"])
print(metric_dictionary)
overall_metrics = pd.DataFrame({
    "指标": ["用户数", "流失人数", "流失率", "平均订单数", "订单数中位数", "平均优惠券数", "平均返现", "平均App时长", "平均满意度", "平均距上次下单天数"],
    "数值": [
        df["CustomerID"].nunique(),
        df["Churn"].sum(),
        df["Churn"].mean(),
        df["OrderCount"].mean(),
        df["OrderCount"].median(),
        df["CouponUsed"].mean(),
        df["CashbackAmount"].mean(),
        df["HourSpendOnApp"].mean(),
        df["SatisfactionScore"].mean(),
        df["DaySinceLastOrder"].mean(),
    ]
})
print(overall_metrics)
print(f"总体流失率：{df['Churn'].mean():.2%}")

profile_fields = ["TenureGroup", "PreferedOrderCat", "PreferredPaymentMode", "PreferredLoginDevice", "CityTier"]
for field in profile_fields:
    table = df[field].value_counts(dropna=False).rename("用户数").to_frame()
    table["用户占比"] = table["用户数"] / len(df)
    print(f"\n--- {field} ---")
    display(table)

tenure_analysis = (
    df.groupby("TenureGroup", observed=True)
      .agg(
          用户数=("CustomerID", "nunique"),
          流失人数=("Churn", "sum"),
          流失率=("Churn", "mean"),
          平均订单数=("OrderCount", "mean"),
          平均返现=("CashbackAmount", "mean"),
          平均距上次下单天数=("DaySinceLastOrder", "mean"),
      )
      .reset_index()
)
display(tenure_analysis)
complain_analysis = (
    df.groupby("Complain")
      .agg(
          用户数=("CustomerID", "nunique"),
          流失人数=("Churn", "sum"),
          流失率=("Churn", "mean"),
          平均满意度=("SatisfactionScore", "mean"),
          平均订单数=("OrderCount", "mean"),
      )
      .reset_index()
)
complain_analysis["投诉状态"] = complain_analysis["Complain"].map({0: "无投诉", 1: "有投诉"})
display(complain_analysis[["投诉状态", "用户数", "流失人数", "流失率", "平均满意度", "平均订单数"]])

category_analysis = (
    df.groupby("PreferedOrderCat")
      .agg(
          用户数=("CustomerID", "nunique"),
          流失率=("Churn", "mean"),
          平均订单数=("OrderCount", "mean"),
          平均优惠券数=("CouponUsed", "mean"),
          平均返现=("CashbackAmount", "mean"),
      )
      .reset_index()
      .sort_values(["流失率", "用户数"], ascending=[False, False])
)
category_analysis["用户占比"] = category_analysis["用户数"] / len(df)
display(category_analysis)

payment_analysis = (
    df.groupby("PreferredPaymentMode")
      .agg(
          用户数=("CustomerID", "nunique"),
          流失率=("Churn", "mean"),
          平均订单数=("OrderCount", "mean"),
          平均优惠券数=("CouponUsed", "mean"),
          平均返现=("CashbackAmount", "mean"),
      )
      .reset_index()
      .sort_values("用户数", ascending=False)
)
display(payment_analysis)

churn_behavior = (
    df.groupby("Churn")
      .agg(
          用户数=("CustomerID", "nunique"),
          平均订单数=("OrderCount", "mean"),
          平均优惠券数=("CouponUsed", "mean"),
          平均返现=("CashbackAmount", "mean"),
          平均App时长=("HourSpendOnApp", "mean"),
          平均满意度=("SatisfactionScore", "mean"),
          平均距上次下单天数=("DaySinceLastOrder", "mean"),
      )
      .reset_index()
)
churn_behavior["用户状态"] = churn_behavior["Churn"].map({0: "未流失", 1: "已流失"})
display(churn_behavior.drop(columns="Churn"))


tenure_complain_analysis = (
    df.groupby(["TenureGroup", "Complain"], observed=True)
      .agg(
          用户数=("CustomerID", "nunique"),
          流失人数=("Churn", "sum"),
          流失率=("Churn", "mean"),
          平均订单数=("OrderCount", "mean"),
      )
      .reset_index()
)
tenure_complain_analysis["投诉状态"] = tenure_complain_analysis["Complain"].map({0: "无投诉", 1: "有投诉"})
tenure_complain_analysis["样本提示"] = np.where(tenure_complain_analysis["用户数"] < 30, "小样本", "可观察")
display(tenure_complain_analysis)



count_pivot = pd.pivot_table(
    df, index="TenureGroup", columns="Complain", values="CustomerID",
    aggfunc="nunique", fill_value=0, observed=True
).rename(columns={0: "无投诉用户数", 1: "有投诉用户数"})

churn_pivot = pd.pivot_table(
    df, index="TenureGroup", columns="Complain", values="Churn",
    aggfunc="mean", observed=True
).rename(columns={0: "无投诉流失率", 1: "有投诉流失率"})

cross_pivot = count_pivot.join(churn_pivot).reset_index()
display(cross_pivot)

outputs = {
    "overall_metrics.csv": overall_metrics,
    "tenure_analysis.csv": tenure_analysis,
    "complain_analysis.csv": complain_analysis,
    "category_analysis.csv": category_analysis,
    "payment_analysis.csv": payment_analysis,
    "tenure_complain_analysis.csv": tenure_complain_analysis,
    "tenure_complain_pivot.csv": cross_pivot,
}

for filename, table in outputs.items():
    path = OUTPUT_DIR / filename
    table.to_csv(path, index=False, encoding="utf-8-sig")
    check = pd.read_csv(path)
    print(f"已输出 {filename}: {check.shape}")