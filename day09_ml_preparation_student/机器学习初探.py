from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

ROOT = Path.cwd()
if ROOT.name == 'notebooks':
    ROOT = ROOT.parent
DATA_PATH = ROOT / 'data/ecommerce_customer_cleaned.csv'
OUTPUT_DIR = ROOT / 'output'
OUTPUT_DIR.mkdir(exist_ok=True)
RANDOM_STATE = 42
TEST_SIZE = 0.20
pd.set_option('display.max_columns', 80)

# TODO 9-0：填写个人信息
STUDENT_NAME = '窦俊磊'
STUDENT_ID = '24012455'
CLASS_NAME = '信计三班'
assert STUDENT_NAME and STUDENT_ID and CLASS_NAME, '请先填写个人信息'

TARGET = 'Churn'
ID_COL = 'CustomerID'


def main():
    toy = pd.DataFrame({
        '用户': list('ABCDEF'),
        'Tenure': [1, 24, 3, 18, 2, 30],
        'Complain': [1, 0, 0, 1, 1, 0],
        'Churn': [1, 0, 1, 0, 1, 0],
    })
    toy['人工规则预测'] = ((toy['Tenure'] <= 3) & (toy['Complain'] == 1)).astype(int)
    toy['是否判断正确'] = toy['人工规则预测'] == toy['Churn']
    print('toy 数据：')
    print(toy)
    print('判断正确：', int(toy['是否判断正确'].sum()), '/', len(toy))
    print(
        '真实流失3人，规则找到：',
        int(((toy['人工规则预测'] == 1) & (toy['Churn'] == 1)).sum()),
        '人',
    )

    df = pd.read_csv(DATA_PATH)
    print('数据形状：', df.shape)
    print('一行代表一名用户；总体流失率：', '{:.2%}'.format(df['Churn'].mean()))
    assert df.shape == (5630, 22)
    assert df['CustomerID'].is_unique
    assert set(df['Churn'].unique()) == {0, 1}
    assert int(df.isna().sum().sum()) == 0

    X = df.drop(columns=[TARGET, ID_COL]).copy()
    y = df[TARGET].astype(int).copy()
    assert TARGET not in X.columns and ID_COL not in X.columns
    print('特征表：', X.shape, '标签：', y.shape)

    categorical_features = X.select_dtypes(include='object').columns.tolist()
    numeric_features = X.select_dtypes(exclude='object').columns.tolist()
    derived_features = ['TenureGroup', 'IsMobileLogin']

    rows = []
    for column in df.columns:
        if column == ID_COL:
            role, action, reason = 'identifier', 'drop', '用户编号只用于追踪'
        elif column == TARGET:
            role, action, reason = 'target', 'separate', '这是希望预测的答案'
        elif column in derived_features:
            role, action, reason = 'derived_feature', 'candidate', '由已有字段转换得到'
        elif column in categorical_features:
            role, action, reason = 'categorical_feature', 'one_hot', '文字类别需要转成0/1列'
        else:
            role, action, reason = 'numeric_feature', 'numeric_pipeline', '进入教师提供的数值处理分支'
        rows.append(
            {
                'feature': column,
                'role': role,
                'dtype': str(df[column].dtype),
                'action': action,
                'reason': reason,
            }
        )

    feature_schema = pd.DataFrame(rows)
    feature_schema.to_csv(OUTPUT_DIR / 'feature_schema.csv', index=False, encoding='utf-8-sig')
    print('已保存 feature_schema.csv')

    STRATIFY_TARGET = y
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=STRATIFY_TARGET
    )
    assert STRATIFY_TARGET is y, '请使用y进行分层划分'

    split_summary = pd.DataFrame(
        [
            {'split': 'train', 'rows': len(X_train), 'churn_count': int(y_train.sum()), 'churn_rate': y_train.mean()},
            {'split': 'test', 'rows': len(X_test), 'churn_count': int(y_test.sum()), 'churn_rate': y_test.mean()},
        ]
    )
    split_summary.to_csv(OUTPUT_DIR / 'split_summary.csv', index=False, encoding='utf-8-sig')
    print(split_summary)
    assert len(X_train) + len(X_test) == 5630
    assert abs(y_train.mean() - y_test.mean()) < 0.01

    numeric_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler()),
    ])
    categorical_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False)),
    ])
    preprocessor = ColumnTransformer(
        [
            ('num', numeric_pipeline, numeric_features),
            ('cat', categorical_pipeline, categorical_features),
        ]
    )

    X_train_ready = preprocessor.fit_transform(X_train)
    X_test_ready = preprocessor.transform(X_test)
    feature_names = preprocessor.get_feature_names_out()

    assert X_train_ready.shape[1] == X_test_ready.shape[1] == 36
    assert np.isfinite(X_train_ready).all() and np.isfinite(X_test_ready).all()

    model_matrix_preview = pd.DataFrame(X_train_ready[:20], columns=feature_names)
    model_matrix_preview.to_csv(OUTPUT_DIR / 'model_matrix_preview.csv', index=False, encoding='utf-8-sig')
    print('训练矩阵：', X_train_ready.shape, '测试矩阵：', X_test_ready.shape)

    baseline = DummyClassifier(strategy='prior', random_state=RANDOM_STATE)
    baseline.fit(X_train_ready, y_train)
    y_pred = baseline.predict(X_test_ready)

    baseline_metrics = pd.DataFrame(
        {
            'metric': ['accuracy', 'churn_recall', 'predicted_churn_count'],
            'value': [
                accuracy_score(y_test, y_pred),
                recall_score(y_test, y_pred, pos_label=1, zero_division=0),
                int(y_pred.sum()),
            ],
        }
    )
    baseline_metrics.to_csv(OUTPUT_DIR / 'baseline_metrics.csv', index=False, encoding='utf-8-sig')
    print(baseline_metrics)
    print('测试集中真实流失人数：', int(y_test.sum()))
    print('最低参照线预测流失人数：', int(y_pred.sum()))

    reflection = (
        '特征是模型可见的输入，如用户属性和行为；标签是希望预测的答案，即 Churn。'
        '训练集用于学习模型规则，测试集用于验证模型在新数据上的表现。'
        '最低参照线是最简单的常数预测，用来判断正式模型是否有改进价值。'
    )
    assert 100 <= len(reflection) <= 200, '请完成100～200字复盘'
    print('复盘：', reflection)


if __name__ == '__main__':
    main()
