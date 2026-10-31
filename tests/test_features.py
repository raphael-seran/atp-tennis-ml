"""Behavior-focused regression tests; require no third-party match data."""
import numpy as np
import pandas as pd
from atp_ml.features import pre_match_features, with_market_probabilities
from atp_ml.train import date_splits, metrics


def matches():
    return pd.DataFrame([
        {'Date':'2025-01-01','Player_1':'A','Player_2':'B','Surface':'Hard','Winner':'A'},
        {'Date':'2025-01-01','Player_1':'A','Player_2':'C','Surface':'Clay','Winner':'C'},
        {'Date':'2025-01-02','Player_1':'A','Player_2':'B','Surface':'Hard','Winner':'B'},
    ])


def test_same_date_does_not_see_another_match_result():
    x=pre_match_features(matches())
    # Both Jan 1 snapshots are formed before either Jan 1 match is applied.
    assert x.loc[0,'Elo_1']==1500
    assert x.loc[1,'Elo_1']==1500
    assert np.isnan(x.loc[1,'Form_1'])
    assert x.loc[2,'H2H_1']==1
    assert x.loc[2,'Elo_1']!=1500


def test_no_future_label_information_in_pre_match_feature():
    df=matches();first=pre_match_features(df)
    df.loc[2,'Winner']='A' # swap a match outcome on the LAST date
    second=pre_match_features(df)
    assert first.loc[2,'Elo_1']==second.loc[2,'Elo_1']
    assert first.loc[2,'H2H_1']==second.loc[2,'H2H_1']


def test_invalid_odds_and_overround():
    x=with_market_probabilities(pd.DataFrame({'Odd_1':[2.,-1.], 'Odd_2':[2.,3.]}))
    assert x.loc[0,'Prob_1']==.5
    assert np.isnan(x.loc[1,'Prob_1'])


def test_chronological_date_boundaries():
    d=pd.DataFrame({'Date':pd.to_datetime(['2025-01-01','2025-01-01','2025-01-02','2025-01-03','2025-01-04','2025-01-05','2025-01-06'])})
    a,b,c=date_splits(d)
    assert a.Date.max()<b.Date.min()<c.Date.min()
    assert len(a)+len(b)+len(c)==len(d)


def test_metrics_are_finite():
    result=metrics([0,1,0,1],[.1,.8,.2,.9])
    assert 0<=result['accuracy']<=1
    assert result['log_loss']>0
