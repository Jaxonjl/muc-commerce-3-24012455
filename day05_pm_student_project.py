from pathlib import Path
import pandas as pd
import numpy as np
try:
    from IPython.display import display
except ImportError:
    def display(obj):
        print(obj)

GROUP_ID = "18"
MEMBERS = ["窦俊磊"]
TOPIC = "ALL"

pd.set_option("display.max_columns", 50)
pd.set_option("display.float_format", lambda x: f"{x:,.2f}")

def find_workspace_root(start=None):
    start = Path.cwd() if start is None else Path(start)
    for candidate in [start, *start.parents]:
        if (candidate / "output" / "day04_project" / "ecommerce_customer_cleaned.csv").exists():
            return candidate
    raise FileNotFoundError("未找到清洗后数据，请检查项目目录。")

ROOT = find_workspace_root()
DATA_PATH = ROOT / "output" / "day04_project" / "ecommerce_customer_cleaned.csv"
OUTPUT_DIR = ROOT / "output" / "day05_student" / GROUP_ID
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

print("小组：", GROUP_ID, MEMBERS)
print("专题：", TOPIC)
print("输入：", DATA_PATH)
print("输出：", OUTPUT_DIR)



# TODO 1：读取清洗后的CSV，变量名必须为df
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


# TODO 2：输出shape、前5行和字段类型
print(df.head())
print(df.shape)
# TODO 3：计算以下验收结果
validation = pd.Series({
    "行数": len(df),
    "列数": df.shape[1],
    "CustomerID重复数": int(df["CustomerID"].duplicated().sum()),
    "核心字段缺失数": int(df[core_cols].isna().sum().sum()),
    "Churn取值": sorted(df["Churn"].unique().tolist()),
}, name="验收结果")
print(validation.to_frame())
# 完成上一个单元后再运行本检查点
assert df.shape == (5630, 22), "数据形状与第4天交付物不一致"
assert df["CustomerID"].is_unique, "CustomerID存在重复"
assert df[core_cols].notna().all().all(), "核心字段仍有缺失"
assert set(df["Churn"].unique()) == {0, 1}, "Churn应只包含0和1"
print("数据验收通过：一行代表一名用户。")


# TODO：构建overall_metrics DataFrame，至少包含以下指标：
# 用户数、流失人数、流失率、平均订单数、订单数中位数、
# 平均优惠券数、平均返现、平均App时长、平均满意度、平均距上次下单天数

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


# 检查点2
assert isinstance(overall_metrics, pd.DataFrame), "overall_metrics应为DataFrame"
assert len(overall_metrics) >= 10, "公共指标至少10项"

# TODO：将下面变量赋值为你计算的总体流失率
overall_churn_rate = df['Churn'].mean()
assert abs(overall_churn_rate - 0.16838365896980462) < 1e-8, "总体流失率不正确"
print("检查点2通过")



# 1. 分组字段（TenureGroup）
segment_field = "TenureGroup"

# 2. 使用groupby + agg完成命名聚合
segment_analysis = (
    df.groupby(segment_field, observed=True)
      .agg(
          用户数=("CustomerID", "nunique"),
          流失人数=("Churn", "sum"),
          流失率=("Churn", "mean"),
          平均订单数=("OrderCount", "mean"),
          平均返现=("CashbackAmount", "mean")
      )
)

# 3. 重置索引、排序
segment_analysis = segment_analysis.reset_index().sort_values("流失率", ascending=False)

# 4. 展示结果（Jupyter用display，VSCode用print）
print(segment_analysis)

# 检查点3
assert segment_field in df.columns, "segment_field不是有效字段"
assert isinstance(segment_analysis, pd.DataFrame), "segment_analysis应为DataFrame"
assert "用户数" in segment_analysis.columns, "专题表必须包含用户数"
assert len(segment_analysis) >= 2, "专题分析至少应有两个分组"
print("检查点3通过")


"""
新用户(0-5 年）对平台的粘性尚未形成，可能因首次体验不佳、竞品优惠或服务感知不足而更容易流失。
老用户的高平均订单数与返现金额可能与长期使用习惯、会员权益或忠诚度计划相关，相关于更低的流失率。
5-10 年用户流失率骤降至11%,可能说明用户在使用5年后已形成平台依赖,流失风险大幅降低。
20 年以上用户流失率仅 1%,需验证是否与专属服务、高返现激励或极高的品牌忠诚度有关。
"""
# 1. 选择两个维度
dim_1 = "TenureGroup"
dim_2 = "Complain"

# 2. 双维度统计
cross_analysis = (
    df.groupby([dim_1, dim_2], observed=True)
      .agg(
          用户数=("CustomerID", "nunique"),
          流失人数=("Churn", "sum"),
          流失率=("Churn", "mean"),
          平均订单数=("OrderCount", "mean")
      )
      .reset_index()
)

# 3. 新增「样本提示」列
cross_analysis["样本提示"] = np.where(cross_analysis["用户数"] < 30, "小样本", "可观察")

# 4. 排序并打印
cross_analysis = cross_analysis.sort_values("流失率", ascending=False)
print(cross_analysis)

# 检查点4
assert dim_1 in df.columns and dim_2 in df.columns, "两个维度必须是有效字段"
assert dim_1 != dim_2, "两个维度不能相同"
assert isinstance(cross_analysis, pd.DataFrame), "cross_analysis应为DataFrame"
assert {"用户数", "流失率", "样本提示"}.issubset(cross_analysis.columns), "双维表缺少必需列"
assert set(cross_analysis["样本提示"]).issubset({"小样本", "可观察"}), "样本提示取值不正确"
print("检查点4通过")


# 定义输出字典（与任务要求文件名完全一致）
outputs = {
    "overall_metrics.csv": overall_metrics,
    "segment_analysis.csv": segment_analysis,
    "cross_analysis.csv": cross_analysis,
}

# 循环导出并回读验证
for filename, table in outputs.items():
    path = OUTPUT_DIR / filename
    # 导出CSV：index=False, encoding="utf-8-sig"
    table.to_csv(path, index=False, encoding="utf-8-sig")
    # 回读并打印shape
    reloaded = pd.read_csv(path)
    print(f"{filename}: 原始shape={table.shape}, 回读shape={reloaded.shape}")


# 检查点5
for filename, table in outputs.items():
    path = OUTPUT_DIR / filename
    assert path.exists(), f"缺少输出文件：{filename}"
    reloaded = pd.read_csv(path)
    assert reloaded.shape == table.shape, f"{filename}回读形状不一致"
print("检查点5通过：三个csv均已成功导出并回读。")


"""
结论 1
平台整体流失率为16.84%，其中0-5年新用户流失问题最为突出，该群体流失率高达35%，远高于5年以上用户，是当前流失防控的核心人群。
结论 2
投诉是极强的流失预警信号。在0-5年新用户中，有投诉用户流失率高达59%，而无投诉用户流失率仅24%，投诉显著放大了新用户的流失风险。
结论 3
用户留存与在网时长呈明显正相关：在网时间越长，流失率越低。20 年以上用户流失率仅1%，同时老用户的平均订单数、平均返现金额均显著高于新用户，用户粘性随使用时间明显增强。
分析限制
1. 本次分析为横截面统计，只能观察特征与流失的相关性，无法确定因果关系，不能直接判定 “投诉” 或 “在网时长” 是流失的直接原因。
2. 部分交叉分组虽然样本量充足，但未进行显著性检验，结论存在一定波动可能。
3. 未纳入时间序列行为变化，无法判断用户是先投诉再流失，还是即将流失才更容易投诉。
运营建议与验证方式
运营建议
针对0-5年有投诉的高风险用户，建立专属客诉快速响应机制，增加回访、补偿与关怀动作，降低该群体流失率。

验证方式
将同类新用户随机分为实验组（加急客诉 + 专属关怀）与对照组（原有流程），持续跟踪30天流失率，对比两组留存差异，验证策略效果
"""


# 1. 使用 qcut 构建订单活跃度分层
# 先看分位数边界，再手动指定标签
df["OrderActiveLevel"] = pd.qcut(
    df["OrderCount"],
    q=3,
    duplicates="drop",  # 自动处理重复边界
    labels=False  # 先不指定标签，避免数量不匹配
).map({0: "低活跃", 1: "中活跃", 2: "高活跃"})  # 手动映射标签

print("订单活跃度分层结果：")
print(df["OrderActiveLevel"].value_counts())

# 2. 设计供第6天绘图使用的长表
long_df = df.melt(
    id_vars=["CustomerID", "Churn", "TenureGroup"],
    value_vars=["OrderCount", "CouponUsed", "CashbackAmount"],
    var_name="BehaviorMetric",
    value_name="MetricValue"
)
print("\n长表结构预览：")
print(long_df.head())

# 3. 对反直觉结果提出两种数据核查方法
print("\n反直觉结果核查（以设备流失差异为例）:")
# 方法1：样本量与分布校验
device_check = df.groupby("PreferredLoginDevice").agg(
    用户数=("CustomerID", "nunique"),
    流失率=("Churn", "mean")
)
print("方法1：设备分组样本量与流失率校验")
print(device_check)

# 方法2：混淆变量控制校验（控制TenureGroup）
device_confound_check = (
    df.groupby(["TenureGroup", "PreferredLoginDevice"], observed=True)
      .agg(
          用户数=("CustomerID", "nunique"),
          流失率=("Churn", "mean")
      )
)
print("\n方法2：控制在网时长后的设备流失率校验")
print(device_confound_check)
