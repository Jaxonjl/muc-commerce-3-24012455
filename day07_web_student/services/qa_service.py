from pathlib import Path

import pandas as pd


def answer_question(base_dir: Path, question: str) -> str:
    data_dir = base_dir / "data"
    metrics_df = pd.read_csv(data_dir / "overall_metrics.csv", encoding="utf-8-sig")
    category_df = pd.read_csv(data_dir / "category_analysis.csv", encoding="utf-8-sig")
    segment_df = pd.read_csv(data_dir / "segment_analysis.csv", encoding="utf-8-sig")
    metrics = dict(zip(metrics_df["指标"], metrics_df["数值"]))
    normalized = question.replace(" ", "").lower()

    if any(word in normalized for word in ["多少用户", "用户数", "总用户", "一共有"]):
        return f"数据集中共有{int(metrics['用户数']):,}名用户。"

    if any(word in normalized for word in ["总体流失率", "整体流失率", "流失率是多少", "流失率"]):
        return (
            f"当前总体流失率为{metrics['流失率']:.1%}，对应流失人数{int(metrics['流失人数']):,}人。"
        )

    if any(word in normalized for word in ["哪个品类", "偏好品类", "品类最多", "用户最多"]):
        top_category = category_df.loc[category_df["用户数"].idxmax()]
        return (
            f"用户数最多的偏好品类是{top_category['PreferedOrderCat']}，共有{int(top_category['用户数']):,}名用户。"
        )

    if any(word in normalized for word in ["生命周期阶段", "风险最高", "哪个阶段", "阶段风险"]):
        top_stage = segment_df.loc[segment_df["流失率"].idxmax()]
        return (
            f"生命周期阶段中，{top_stage['TenureGroup']}的流失率最高，为{top_stage['流失率']:.1%}。"
        )

    if any(word in normalized for word in ["平均订单数", "订单情况", "订单平均", "订单数"]):
        return (
            f"整体平均订单数为{metrics['平均订单数']:.2f}单，中位数为{int(metrics['订单数中位数']):,}单。"
        )

    return (
        "抱歉，此问题超出当前项目规则问答范围。"
        "目前支持总体规模、流失情况、偏好品类、生命周期风险和订单情况。"
    )
