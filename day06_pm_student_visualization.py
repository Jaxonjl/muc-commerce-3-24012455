from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter

try:
    from IPython.display import display
except ImportError:
    def display(obj):
        print(obj)

STUDENT_ID = "24012455"
TOPIC = "ALL"

pd.set_option("display.max_columns", 50)
pd.set_option("display.float_format", lambda x: f"{x:,.2f}")
plt.rcParams["font.sans-serif"] = [
    "Microsoft YaHei", "SimHei", "PingFang SC",
    "Heiti SC", "Arial Unicode MS", "DejaVu Sans",
]
plt.rcParams["axes.unicode_minus"] = False

def find_workspace_root(start=None):
    start = Path.cwd() if start is None else Path(start)
    for candidate in [start, *start.parents]:
        if (candidate / "output" / "day04_project" / "ecommerce_customer_cleaned.csv").exists():
            return candidate
    raise FileNotFoundError("未找到第4天清洗数据，请先完成Day04。")


ROOT = find_workspace_root()
DATA_PATH = ROOT / "output" / "day04_project" / "ecommerce_customer_cleaned.csv"
DAY05_DIR = ROOT / "output" / "day05_analysis"
OUTPUT_DIR = ROOT / "output" / "day06_visualization"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

print("学生：", STUDENT_ID)
print("专题：", TOPIC)
print("输出：", OUTPUT_DIR.relative_to(ROOT))

required_inputs = [
    DATA_PATH,
    DAY05_DIR / "overall_metrics.csv",
    DAY05_DIR / "segment_analysis.csv",
    DAY05_DIR / "cross_analysis.csv",
]
missing_inputs = [str(path.relative_to(ROOT)) for path in required_inputs if not path.exists()]
assert not missing_inputs, f"缺少输入文件：{missing_inputs}"

df = pd.read_csv(DATA_PATH)
overall_metrics = pd.read_csv(required_inputs[1])
segment_analysis = pd.read_csv(required_inputs[2])
cross_analysis = pd.read_csv(required_inputs[3])

assert df.shape[0] == 5630, f"清洗数据行数异常：{df.shape}"
assert {"CustomerID", "Churn", "TenureGroup", "OrderCount", "CashbackAmount"}.issubset(df.columns)
assert set(df["Churn"].dropna().unique()).issubset({0, 1})

display(overall_metrics)
display(segment_analysis.head())
display(cross_analysis.head())
print("检查点1A通过：输入文件有效")

# TODO：填写4个业务问题和图表选择理由
business_questions = {
    "category_bar": "各商品类别的订单数量和销售额哪个最高？",
    "behavior_scatter": "客户活跃度与累计消费是否存在关联？",
    "ordered_line": "不同用户级别或时间段的订单趋势如何变化？",
    "composition_chart": "不同渠道/客户类型的用户构成比例是多少？",
}

chart_reasons = {
    "category_bar": "条形图适合对比不同类别之间的数值大小。",
    "behavior_scatter": "散点图能展示两个指标之间的相关性和分布情况。",
    "ordered_line": "折线图适合展示随时间或排序变化的趋势。",
    "composition_chart": "构成图可以清晰展示各部分占整体的比例。",
}

assert all(text.strip() for text in business_questions.values()), "请填写4个业务问题"
assert all(text.strip() for text in chart_reasons.values()), "请填写4个图表选择理由"
print("检查点1B通过：业务问题和选择理由已填写")


# TODO：完成绘图数据。建议使用自己的第5天主分组字段。
category_field = "PreferedOrderCat"

category_summary = (
    df.groupby(category_field, observed=True)
      .agg(
          用户数=("CustomerID", "nunique"),
          流失率=("Churn", "mean"),
          平均返现=("CashbackAmount", "mean"),
      )
      .reset_index()
      .sort_values("用户数", ascending=False)
)

assert category_field in df.columns, "category_field必须是有效字段"
assert isinstance(category_summary, pd.DataFrame), "category_summary必须是DataFrame"
assert {category_field, "用户数"}.issubset(category_summary.columns)
display(category_summary)

fig_bar, ax_bar = plt.subplots(figsize=(12, 6))

bars = ax_bar.bar(
    category_summary[category_field],
    category_summary["用户数"],
    color="#5B8FF9",
    alpha=0.8,
    label="用户数"
)
ax_bar.set_xlabel(category_field)
ax_bar.set_ylabel("用户数")
ax_bar.set_title("偏好品类用户规模与流失率对比")
ax_bar.tick_params(axis="x", rotation=30)

for bar, rate in zip(bars, category_summary["流失率"]):
    height = bar.get_height()
    ax_bar.text(bar.get_x() + bar.get_width() / 2, height + 10,
                f"{height:.0f}\n{rate:.1%}",
                ha="center", va="bottom", fontsize=9)

ax_bar.grid(axis="y", linestyle="--", alpha=0.3)

bar_path = OUTPUT_DIR / "01_category_bar.png"
fig_bar.savefig(bar_path, dpi=150, bbox_inches="tight")
plt.close(fig_bar)

assert bar_path.exists() and bar_path.stat().st_size > 0, "柱状图尚未保存"
print("已输出：", bar_path.relative_to(ROOT))

x_field = "OrderCount"
y_field = "CashbackAmount"

assert x_field in df.columns and y_field in df.columns
assert pd.api.types.is_numeric_dtype(df[x_field])
assert pd.api.types.is_numeric_dtype(df[y_field])

fig_scatter, ax_scatter = plt.subplots(figsize=(12, 6))

for churn_value, label, color in [(0, "未流失", "#5B8FF9"), (1, "已流失", "#F76E76")]:
    subset = df[df["Churn"] == churn_value]
    ax_scatter.scatter(
        subset[x_field],
        subset[y_field],
        alpha=0.5,
        label=label,
        s=30,
        color=color
    )

ax_scatter.set_xlabel("OrderCount")
ax_scatter.set_ylabel("CashbackAmount")
ax_scatter.set_title("OrderCount 与 CashbackAmount 的用户行为散点图")
ax_scatter.legend(title="Churn")
ax_scatter.grid(True, linestyle="--", alpha=0.3)

scatter_path = OUTPUT_DIR / "02_behavior_scatter.png"
fig_scatter.savefig(scatter_path, dpi=150, bbox_inches="tight")
plt.close(fig_scatter)

assert scatter_path.exists() and scatter_path.stat().st_size > 0, "散点图尚未保存"
print("已输出：", scatter_path.relative_to(ROOT))

TENURE_ORDER = ["新用户", "0-6个月", "7-12个月", "13-24个月", "24个月以上"]
ordered_field = "TenureGroup"

ordered_summary = (
    df.groupby(ordered_field, observed=True)
      .agg(
          用户数=("CustomerID", "nunique"),
          流失率=("Churn", "mean"),
      )
      .reset_index()
)

ordered_summary[ordered_field] = pd.Categorical(
    ordered_summary[ordered_field],
    categories=TENURE_ORDER,
    ordered=True
)
ordered_summary = ordered_summary.sort_values(ordered_field)

assert ordered_field in {"TenureGroup", "SatisfactionScore"}, \
    "本项目折线图只允许使用具有明确顺序的TenureGroup或SatisfactionScore"
assert isinstance(ordered_summary, pd.DataFrame)
assert {ordered_field, "用户数"}.issubset(ordered_summary.columns)
display(ordered_summary)

fig_line, ax_line = plt.subplots(figsize=(12, 6))

ax_line.plot(
    ordered_summary[ordered_field],
    ordered_summary["流失率"],
    marker="o",
    color="#FA8C16",
    linewidth=2,
    label="流失率"
)
ax_line.set_xlabel(ordered_field)
ax_line.set_ylabel("流失率")
ax_line.set_title("不同 TenureGroup 的流失率阶段比较")
ax_line.set_ylim(0, ordered_summary["流失率"].max() * 1.2)
ax_line.grid(True, linestyle="--", alpha=0.3)

for x, y, n in zip(ordered_summary[ordered_field], ordered_summary["流失率"], ordered_summary["用户数"]):
    ax_line.text(x, y + 0.005, f"{y:.1%}\nN={int(n)}", ha="center", va="bottom", fontsize=9)

line_path = OUTPUT_DIR / "03_ordered_line.png"
fig_line.savefig(line_path, dpi=150, bbox_inches="tight")
plt.close(fig_line)

assert line_path.exists() and line_path.stat().st_size > 0, "折线图尚未保存"
print("已输出：", line_path.relative_to(ROOT))


composition_field = "CityTier"

composition_summary = (
    df.groupby(composition_field, observed=True)
      .agg(用户数=("CustomerID", "nunique"))
      .reset_index()
)

composition_summary["占比"] = composition_summary["用户数"] / composition_summary["用户数"].sum()
composition_summary = composition_summary.sort_values("用户数", ascending=False)

assert composition_field in df.columns
assert isinstance(composition_summary, pd.DataFrame)
assert {composition_field, "用户数", "占比"}.issubset(composition_summary.columns)
assert np.isclose(composition_summary["占比"].sum(), 1.0), "构成占比之和应为1"
display(composition_summary)


fig_composition, ax_composition = plt.subplots(figsize=(10, 6))

if len(composition_summary) <= 5:
    wedges, texts, autotexts = ax_composition.pie(
        composition_summary["用户数"],
        labels=composition_summary[composition_field],
        autopct="%.1f%%",
        startangle=140,
        textprops={"fontsize": 10}
    )
    ax_composition.axis("equal")
    ax_composition.set_title("CityTier 用户占比")
else:
    ax_composition.bar(
        composition_summary[composition_field],
        composition_summary["用户数"],
        color="#73D13D",
        alpha=0.8
    )
    ax_composition.set_xlabel(composition_field)
    ax_composition.set_ylabel("用户数")
    ax_composition.set_title("CityTier 用户构成柱状图")
    ax_composition.tick_params(axis="x", rotation=30)

composition_path = OUTPUT_DIR / "04_composition_chart.png"
fig_composition.savefig(composition_path, dpi=150, bbox_inches="tight")
plt.close(fig_composition)

assert composition_path.exists() and composition_path.stat().st_size > 0, "构成图尚未保存"
print("已输出：", composition_path.relative_to(ROOT))

fig_summary, axes = plt.subplots(2, 2, figsize=(14, 10))
axes = axes.flatten()

# 1. 类别柱状图
axes[0].bar(
    category_summary[category_field],
    category_summary["用户数"],
    color="#5B8FF9",
    alpha=0.8
)
axes[0].set_title("偏好品类用户规模")
axes[0].tick_params(axis="x", rotation=30)

# 2. 行为散点图
for churn_value, label, color in [(0, "未流失", "#5B8FF9"), (1, "已流失", "#F76E76")]:
    subset = df[df["Churn"] == churn_value]
    axes[1].scatter(
        subset[x_field],
        subset[y_field],
        alpha=0.5,
        label=label,
        s=20,
        color=color
    )
axes[1].set_title("订单数与返现金额散点图")
axes[1].set_xlabel(x_field)
axes[1].set_ylabel(y_field)
axes[1].legend()

# 3. 折线图
axes[2].plot(
    ordered_summary[ordered_field],
    ordered_summary["流失率"],
    marker="o", color="#FA8C16"
)
axes[2].set_title("TenureGroup 阶段流失率")
axes[2].set_ylabel("流失率")
axes[2].grid(True, linestyle="--", alpha=0.3)

# 4. 构成图
if len(composition_summary) <= 5:
    axes[3].pie(
        composition_summary["用户数"],
        labels=composition_summary[composition_field],
        autopct="%.1f%%",
        startangle=140
    )
    axes[3].axis("equal")
    axes[3].set_title("CityTier 构成")
else:
    axes[3].bar(
        composition_summary[composition_field],
        composition_summary["用户数"],
        color="#73D13D"
    )
    axes[3].set_title("CityTier 构成")

fig_summary.suptitle("电商用户数据可视化分析概览", fontsize=16, fontweight="bold")
fig_summary.tight_layout(rect=[0, 0, 1, 0.96])

summary_path = OUTPUT_DIR / "day06_visualization_summary.png"
fig_summary.savefig(summary_path, dpi=150, bbox_inches="tight")
plt.close(fig_summary)

assert summary_path.exists() and summary_path.stat().st_size > 0, "综合图尚未保存"
print("已输出：", summary_path.relative_to(ROOT))


chart_manifest = pd.DataFrame([
    {
        "chart_id": "01",
        "file_name": "01_category_bar.png",
        "business_question": business_questions["category_bar"],
        "chart_type": "bar",
        "key_finding": "偏好品类用户数差异显著，用户规模大的品类流失率相对较低。",
        "limitation": "该图只能说明品类间差异，不能解释原因。"
    },
    {
        "chart_id": "02",
        "file_name": "02_behavior_scatter.png",
        "business_question": business_questions["behavior_scatter"],
        "chart_type": "scatter",
        "key_finding": "高订单数用户的返现金额总体更高，已流失用户分布较散。",
        "limitation": "散点图仅展示相关性，不证明因果关系。"
    },
    {
        "chart_id": "03",
        "file_name": "03_ordered_line.png",
        "business_question": business_questions["ordered_line"],
        "chart_type": "line",
        "key_finding": "TenureGroup 流失率呈现阶段性变化，新用户和老阶段流失率较高。",
        "limitation": "TenureGroup 是阶段，不等同于连续时间序列。"
    },
    {
        "chart_id": "04",
        "file_name": "04_composition_chart.png",
        "business_question": business_questions["composition_chart"],
        "chart_type": "pie_or_bar",
        "key_finding": "CityTier 构成显示中高线城市用户占比较大。",
        "limitation": "用户构成不等于贡献度，不能直接说明营收结构。"
    },
    {
        "chart_id": "05",
        "file_name": "day06_visualization_summary.png",
        "business_question": "整体概览",
        "chart_type": "dashboard",
        "key_finding": "从4张图一起看，用户规模、流失率、订单行为和城市构成形成了完整画像。",
        "limitation": "综合图为概览，不适合查看单图的精细数据。"
    },
])

assert len(chart_manifest) == 5
assert not chart_manifest.astype(str).apply(lambda col: col.str.contains("请填写").any()).any(), \
    "请完成图表清单"

manifest_path = OUTPUT_DIR / "chart_manifest.csv"
chart_manifest.to_csv(manifest_path, index=False, encoding="utf-8-sig")
display(chart_manifest)