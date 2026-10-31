"""Reproducible core benchmark for ATP match prediction (CSV supplied by user).

Unlike the team's original exploratory feature search, this script keeps the
final test dates untouched by model selection: train < validation < test.
Results can legitimately differ from those in the 2026 project report.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, log_loss
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .features import pre_match_features, with_market_probabilities

NUMERIC = [
    "Rank_1", "Rank_2", "Rank_Diff", "Pts_1", "Pts_2", "Pts_Diff",
    "Elo_1", "Elo_2", "Elo_Diff", "Elo_Surf_1", "Elo_Surf_2",
    "Elo_Surf_Diff", "Form_1", "Form_2", "Form_Diff", "H2H_1",
    "H2H_2", "H2H_Diff", "Best of",
]
CATEGORICAL = ["Surface", "Series", "Court"]


def date_splits(df: pd.DataFrame, train_frac: float = .6, valid_frac: float = .2):
    """Split on *unique dates*, never separating the same day's matches."""
    dates = np.sort(pd.to_datetime(df["Date"]).unique())
    if len(dates) < 6:
        raise ValueError("Need at least six distinct dates for train/validation/test")
    t = max(1,int(len(dates)*train_frac))
    v = max(t+1,int(len(dates)*(train_frac+valid_frac)))
    if v >= len(dates): raise ValueError("Split leaves no test dates")
    return df[df.Date < dates[t]], df[(df.Date >= dates[t])&(df.Date < dates[v])], df[df.Date >= dates[v]]


def metrics(y, probability):
    predicted = np.asarray(probability)>=.5
    y=np.asarray(y)
    return {"accuracy": float(accuracy_score(y,predicted)),
            "f1": float(f1_score(y,predicted,zero_division=0)),
            "roc_auc": float(roc_auc_score(y,probability)) if len(set(y))==2 else None,
            "log_loss": float(log_loss(y,probability,labels=[0,1]))}


def prepare_data(path):
    df=pd.read_csv(path)
    df["Date"]=pd.to_datetime(df["Date"],errors="raise")
    for c in ("Rank_1","Rank_2","Pts_1","Pts_2","Best of"):
        df[c]=pd.to_numeric(df[c],errors="coerce")
        df.loc[df[c]<0,c]=np.nan # original dataset uses -1 for missing values
    df=pre_match_features(df)
    df=with_market_probabilities(df)
    df["Rank_Diff"]=df["Rank_1"]-df["Rank_2"]
    df["Pts_Diff"]=df["Pts_1"]-df["Pts_2"]
    # One common sample for all models AND bookmaker baseline.
    return df.dropna(subset=["Prob_1","Rank_1","Rank_2","Pts_1","Pts_2","Form_1","Form_2"]).copy()


def make_model(estimator, use_market=False):
    cols=NUMERIC+(["Prob_1"] if use_market else [])
    pre=ColumnTransformer([
        ("num",Pipeline([("imputer",SimpleImputer(strategy="median")),("scale",StandardScaler())]),cols),
        ("categorical",OneHotEncoder(handle_unknown="ignore",sparse_output=False),CATEGORICAL),
    ])
    return Pipeline([("preprocess",pre),("model",estimator)])


def run(csv_path, include_xgboost=True):
    df=prepare_data(csv_path)
    train,valid,test=date_splits(df)
    if min(len(x) for x in [train,valid,test])<2: raise ValueError("Not enough eligible examples in each split")
    features=NUMERIC+CATEGORICAL+['Prob_1']
    candidates={
        "Logistic regression (sports only)":make_model(LogisticRegression(max_iter=1000),False),
        "Random forest (sports only)":make_model(RandomForestClassifier(n_estimators=120,max_depth=10,random_state=42,n_jobs=-1),False),
    }
    if include_xgboost:
        try:
            from xgboost import XGBClassifier
            candidates["XGBoost (sports only)"]=make_model(XGBClassifier(n_estimators=200,max_depth=5,learning_rate=.05,subsample=.8,colsample_bytree=.8,random_state=42,eval_metric='logloss',n_jobs=2),False)
            candidates["XGBoost + bookmaker odds"]=make_model(XGBClassifier(n_estimators=200,max_depth=5,learning_rate=.05,subsample=.8,colsample_bytree=.8,random_state=42,eval_metric='logloss',n_jobs=2),True)
        except ImportError:
            print('xgboost not available; using sklearn candidates only')
    # Validation is used for selection; final test remains untouched.
    validation={}
    for name, model in candidates.items():
        model.fit(train[features],train.Target)
        p=model.predict_proba(valid[features])[:,1]
        validation[name]=metrics(valid.Target,p)
    chosen=min(validation,key=lambda k: validation[k]['log_loss'])
    best=candidates[chosen]
    # Fit on train+validation, evaluate once on chronological test.
    combined=pd.concat([train,valid],axis=0).sort_values('Date',kind='stable')
    best.fit(combined[features],combined.Target)
    result=metrics(test.Target,best.predict_proba(test[features])[:,1])
    return {
        'source':str(csv_path),
        'note':'New clean-pipeline evaluation; NOT the original team presentation result',
        'n_rows':len(df),
        'date_ranges':{x:{'start':str(y.Date.min().date()),'end':str(y.Date.max().date()),'n':len(y)} for x,y in [('train',train),('validation',valid),('test',test)]},
        'validation_metrics':validation,
        'chosen_by_validation_log_loss':chosen,
        'held_out_test_metrics':result,
        'held_out_bookmaker_metrics':metrics(test.Target,test.Prob_1),
    }


def main():
    ap=argparse.ArgumentParser(description='Leakage-aware ATP tennis benchmark')
    ap.add_argument('--csv',required=True,help='Path to licensed atp_tennis.csv')
    ap.add_argument('--out',default='outputs/metrics.json')
    ap.add_argument('--skip-xgboost',action='store_true')
    args=ap.parse_args()
    result=run(args.csv,include_xgboost=not args.skip_xgboost)
    dest=Path(args.out);dest.parent.mkdir(parents=True,exist_ok=True)
    dest.write_text(json.dumps(result,indent=2,ensure_ascii=False)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k != 'validation_metrics'},indent=2))
    print('Saved:',dest)

if __name__=='__main__':main()
