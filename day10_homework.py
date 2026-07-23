from pathlib import Path
import json
import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier

RANDOM_STATE = 42
TEST_SIZE = 0.20

STUDENT_NAME = '窦俊磊'
STUDENT_ID = '24012455'
CLASS_NAME = '信计三班'
assert STUDENT_NAME and STUDENT_ID and CLASS_NAME, '请先填写个人信息'

PROJECT_ROOT = Path.cwd()
DATA_PATH = PROJECT_ROOT / 'data' / 'ecommerce_customer_cleaned.csv'
OUTPUT_DIR = PROJECT_ROOT / 'output'
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

TARGET = 'Churn'
ID_COL = 'CustomerID'

assert DATA_PATH.exists(), f'找不到数据文件：{DATA_PATH}'


def build_preprocessor(numeric_features, categorical_features):
    numeric_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler()),
    ])
    categorical_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False)),
    ])
    return ColumnTransformer([
        ('num', numeric_pipeline, numeric_features),
        ('cat', categorical_pipeline, categorical_features),
    ])


def build_pipeline(model, numeric_features, categorical_features):
    return Pipeline([
        ('preprocessor', build_preprocessor(numeric_features, categorical_features)),
        ('model', model),
    ])


def metric_row(y_true, pred, model_name):
    tn, fp, fn, tp = confusion_matrix(y_true, pred, labels=[0, 1]).ravel()
    return {
        'model': model_name,
        'accuracy': accuracy_score(y_true, pred),
        'precision': precision_score(y_true, pred, zero_division=0),
        'churn_recall': recall_score(y_true, pred, zero_division=0),
        'predicted_churn_count': int(pred.sum()),
        'tn': int(tn),
        'fp': int(fp),
        'fn': int(fn),
        'tp': int(tp),
    }


def choose_final_model(model_comparison):
    candidates = model_comparison[model_comparison['model'].isin([
        'logistic_regression', 'decision_tree', 'random_forest'])].copy()
    candidates = candidates.sort_values(
        by=['churn_recall', 'precision', 'accuracy'],
        ascending=[False, False, False],
        ignore_index=True,
    )
    return candidates.loc[0, 'model']


def make_selection_text(model_name, comparison_df):
    return (
        f'最终选择{model_name}作为业务筛查模型，原因是它在同一测试集上表现出最优的流失召回率，'
        f'能够更稳定地发现潜在流失客户。同时，该模型的精确率和准确率也处于可接受范围，'
        f'在业务筛查中具有较好召回与误报之间的平衡，便于后续重点客户维护。'
    )


def make_reflection_text():
    return (
        '本次复盘说明了最低参照线只能作为基本参考，不能替代真实模型，因为它仅根据标签分布预测，缺少特征信息。'
        '三个模型必须在同一测试集上公平比较，才能保证评估指标一致，避免模型性能因数据划分差异而产生误导。'
        '最终模型的选择要综合关注流失召回率、精确率和漏报/误报人数，'
        '这样才能在业务筛查中既提升发现潜在流失客户的能力，又避免浪费客服资源进行过多错误干预。'
    )


def main():
    df = pd.read_csv(DATA_PATH)
    print('数据形状：', df.shape)
    print('总体流失率：', f"{df['Churn'].mean():.2%}")
    assert df.shape == (5630, 22)
    assert df['CustomerID'].is_unique
    assert set(df['Churn'].unique()) == {0, 1}
    assert df.isna().sum().sum() == 0

    X = df.drop(columns=[TARGET, ID_COL]).copy()
    y = df[TARGET].astype(int).copy()
    customer_ids = df[ID_COL].copy()
    assert TARGET not in X.columns and ID_COL not in X.columns
    print('特征数：', X.shape[1], '标签流失人数：', int(y.sum()))

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y)
    test_customer_ids = customer_ids.loc[X_test.index]
    print('训练集：', X_train.shape, f"流失率={y_train.mean():.2%}")
    print('测试集：', X_test.shape, f"流失率={y_test.mean():.2%}")
    assert len(X_train) == 4504 and len(X_test) == 1126
    assert abs(y_train.mean() - y_test.mean()) < 0.001

    categorical_features = X.select_dtypes(include=['object', 'string']).columns.tolist()
    numeric_features = X.columns.difference(categorical_features).tolist()

    fitted_models = {}
    predictions = {}
    probabilities = {}

    logistic_pipeline = build_pipeline(
        LogisticRegression(max_iter=1000, class_weight='balanced', random_state=RANDOM_STATE),
        numeric_features,
        categorical_features,
    )
    logistic_pipeline.fit(X_train, y_train)
    fitted_models['logistic_regression'] = logistic_pipeline
    predictions['logistic_regression'] = logistic_pipeline.predict(X_test)
    probabilities['logistic_regression'] = logistic_pipeline.predict_proba(X_test)[:, 1]
    print('逻辑回归训练完成；预测流失人数：', int(predictions['logistic_regression'].sum()))

    tree_pipeline = build_pipeline(
        DecisionTreeClassifier(max_depth=5, min_samples_leaf=20,
                               class_weight='balanced', random_state=RANDOM_STATE),
        numeric_features,
        categorical_features,
    )
    tree_pipeline.fit(X_train, y_train)
    fitted_models['decision_tree'] = tree_pipeline
    predictions['decision_tree'] = tree_pipeline.predict(X_test)
    probabilities['decision_tree'] = tree_pipeline.predict_proba(X_test)[:, 1]
    print('决策树训练完成；预测流失人数：', int(predictions['decision_tree'].sum()))

    forest_pipeline = build_pipeline(
        RandomForestClassifier(
            n_estimators=100, max_depth=8, min_samples_leaf=10,
            class_weight='balanced', random_state=RANDOM_STATE, n_jobs=-1),
        numeric_features,
        categorical_features,
    )
    forest_pipeline.fit(X_train, y_train)
    fitted_models['random_forest'] = forest_pipeline
    predictions['random_forest'] = forest_pipeline.predict(X_test)
    probabilities['random_forest'] = forest_pipeline.predict_proba(X_test)[:, 1]
    print('随机森林训练完成；预测流失人数：', int(predictions['random_forest'].sum()))

    baseline_pipeline = build_pipeline(
        DummyClassifier(strategy='prior', random_state=RANDOM_STATE),
        numeric_features,
        categorical_features,
    )
    baseline_pipeline.fit(X_train, y_train)
    fitted_models['baseline'] = baseline_pipeline
    predictions['baseline'] = baseline_pipeline.predict(X_test)
    probabilities['baseline'] = baseline_pipeline.predict_proba(X_test)[:, 1]

    model_order = ['baseline', 'logistic_regression', 'decision_tree', 'random_forest']
    model_comparison = pd.DataFrame([metric_row(y_test, predictions[name], name)
                                     for name in model_order])
    model_comparison.to_csv(OUTPUT_DIR / 'model_comparison.csv', index=False)
    print(model_comparison.to_string(index=False, formatters={
        'accuracy': '{:.2%}'.format,
        'precision': '{:.2%}'.format,
        'churn_recall': '{:.2%}'.format,
    }))

    confusion_summary = model_comparison[['model', 'tn', 'fp', 'fn', 'tp']].copy()
    confusion_summary['total'] = confusion_summary[['tn', 'fp', 'fn', 'tp']].sum(axis=1)
    confusion_summary.to_csv(OUTPUT_DIR / 'confusion_matrix_summary.csv', index=False)
    assert (confusion_summary['total'] == len(y_test)).all()
    print('\n混淆矩阵汇总：')
    print(confusion_summary.to_string(index=False))

    SELECTED_MODEL_NAME = choose_final_model(model_comparison)
    assert SELECTED_MODEL_NAME in {'logistic_regression', 'decision_tree', 'random_forest'}
    selected_pipeline = fitted_models[SELECTED_MODEL_NAME]
    selected_prediction = predictions[SELECTED_MODEL_NAME]
    selected_probability = probabilities[SELECTED_MODEL_NAME]
    print('\n最终模型：', SELECTED_MODEL_NAME)

    selection_note = make_selection_text(SELECTED_MODEL_NAME, model_comparison)
    assert 80 <= len(selection_note) <= 180, '请完成80～180字模型选择说明'
    (OUTPUT_DIR / 'model_selection_note.txt').write_text(selection_note, encoding='utf-8')
    print('\n模型选择说明：')
    print(selection_note)

    customer_predictions = pd.DataFrame({
        'CustomerID': test_customer_ids.to_numpy(),
        'actual_churn': y_test.to_numpy(),
        'predicted_churn': selected_prediction.astype(int),
        'churn_probability': selected_probability,
    })
    customer_predictions['prediction_correct'] = (
        customer_predictions['actual_churn'] == customer_predictions['predicted_churn'])
    customer_predictions.to_csv(OUTPUT_DIR / 'customer_churn_predictions.csv', index=False)
    print('\n客户预测前5行：')
    print(customer_predictions.head().to_string(index=False))
    assert len(customer_predictions) == 1126
    assert customer_predictions['CustomerID'].is_unique

    high_risk_customers = (
        customer_predictions.query('predicted_churn == 1')
        .sort_values('churn_probability', ascending=False)
        .reset_index(drop=True)
    )
    high_risk_customers.to_csv(OUTPUT_DIR / 'high_risk_customers.csv', index=False)
    print('进入优先关注名单的人数：', len(high_risk_customers))
    print(high_risk_customers.head(10).to_string(index=False))

    preprocessor = selected_pipeline.named_steps['preprocessor']
    model = selected_pipeline.named_steps['model']
    feature_names = preprocessor.get_feature_names_out()
    if hasattr(model, 'feature_importances_'):
        importance_values = model.feature_importances_
    elif hasattr(model, 'coef_'):
        importance_values = np.abs(model.coef_[0])
    else:
        importance_values = np.zeros(len(feature_names))
    feature_importance = (pd.DataFrame({
        'feature': feature_names,
        'importance': importance_values,
    }).sort_values('importance', ascending=False).reset_index(drop=True))
    feature_importance.to_csv(OUTPUT_DIR / 'feature_importance.csv', index=False)
    print('\n前10个重要特征：')
    print(feature_importance.head(10).to_string(index=False))

    MODEL_PATH = OUTPUT_DIR / 'selected_model.joblib'
    joblib.dump(selected_pipeline, MODEL_PATH)
    reloaded_pipeline = joblib.load(MODEL_PATH)
    reloaded_prediction = reloaded_pipeline.predict(X_test)
    assert np.array_equal(reloaded_prediction, selected_prediction)
    metadata = {
        'selected_model': SELECTED_MODEL_NAME,
        'random_state': RANDOM_STATE,
        'test_rows': len(X_test),
        'feature_columns': X.columns.tolist(),
    }
    (OUTPUT_DIR / 'model_metadata.json').write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding='utf-8')
    print('模型已保存并通过重新加载检查：', MODEL_PATH)

    reflection = make_reflection_text()
    assert 150 <= len(reflection) <= 250, '请完成150～250字复盘'
    (OUTPUT_DIR / 'reflection.txt').write_text(reflection, encoding='utf-8')
    print('\n学习复盘：')
    print(reflection)

    required = {
        'model_comparison.csv', 'confusion_matrix_summary.csv',
        'customer_churn_predictions.csv', 'high_risk_customers.csv',
        'feature_importance.csv', 'selected_model.joblib',
        'model_metadata.json', 'model_selection_note.txt', 'reflection.txt',
    }
    actual = {path.name for path in OUTPUT_DIR.iterdir() if path.is_file()}
    missing = required - actual
    print('成果文件：', sorted(actual))
    assert not missing, f'缺少成果文件：{sorted(missing)}'
    print('第10天Notebook检查通过')


if __name__ == '__main__':
    main()
