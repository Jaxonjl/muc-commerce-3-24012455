from pathlib import Path

import pandas as pd


def _read_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, encoding="utf-8-sig")


def load_dashboard_data(base_dir: Path, selected_category: str = "全部") -> dict:
    data_dir = base_dir / "data"
    metrics_df = _read_csv(data_dir / "overall_metrics.csv")
    category_df = _read_csv(data_dir / "category_analysis.csv")
    segment_df = _read_csv(data_dir / "segment_analysis.csv")

    metric_map = dict(zip(metrics_df["指标"], metrics_df["数值"]))
    metrics = [
        {"label": "总用户数", "value": f"{int(metric_map['用户数']):,}", "note": "人"},
        {"label": "流失用户", "value": f"{int(metric_map['流失人数']):,}", "note": "人"},
        {"label": "总体流失率", "value": f"{metric_map['流失率']:.1%}", "note": "%"},
        {"label": "平均订单数", "value": f"{metric_map['平均订单数']:.2f}", "note": "单"},
    ]

    categories = ["全部", *category_df["PreferedOrderCat"].tolist()]
    table_df = category_df.copy()
    if selected_category != "全部":
        table_df = table_df[table_df["PreferedOrderCat"] == selected_category]

    table_df = table_df.rename(
        columns={
            "PreferedOrderCat": "偏好品类",
            "用户数": "用户数",
            "流失率": "流失率",
            "平均订单数": "平均订单数",
        }
    )[["偏好品类", "用户数", "流失率", "平均订单数"]]
    table_df["流失率"] = table_df["流失率"].map(lambda value: f"{value:.1%}")
    table_df["平均订单数"] = table_df["平均订单数"].map(lambda value: f"{value:.2f}")

    segment_df = _read_csv(data_dir / "segment_analysis.csv")
    top_stage = segment_df.loc[segment_df["流失率"].idxmax()]
    stage_name = str(top_stage["TenureGroup"])
    stage_rate = top_stage["流失率"]
    if "新用户" in stage_name:
        boundary = "首月留存风险最高"
    elif "0-6" in stage_name:
        boundary = "0-6个月是关键流失窗口"
    elif "7-12" in stage_name:
        boundary = "7-12个月阶段需要持续关注"
    elif "13-24" in stage_name:
        boundary = "13-24个月阶段仍有一定流失"
    else:
        boundary = "后期用户留存较为稳健"
    insight = (
        f"生命周期阶段中，{stage_name}的流失率最高，为{stage_rate:.1%}，{boundary}。"
    )

    return {
        "metrics": metrics,
        "categories": categories,
        "category_rows": table_df.to_dict("records"),
        "insight": insight,
    }
