
import numpy as np
import pandas as pd


col =['職場の様子', '休日休暇(月曜日)', '大手企業', '（派遣先）配属先部署　人数', '残業月20時間以上',
       '1日7時間以下勤務OK', '短時間勤務OK(1日4h以内)', 'Wordのスキルを活かす', '大量募集',
       'Accessのスキルを活かす', '休日休暇(火曜日)', '平日休みあり', '期間・時間　勤務期間', '週2・3日OK',
       '勤務先公開', 'Excelのスキルを活かす', '16時前退社OK', '残業月20時間未満', '英語力不要',
       '英語以外の語学力を活かす', '休日休暇(祝日)', '外資系企業', 'PowerPointのスキルを活かす', '休日休暇(水曜日)',
       '仕事の仕方', 'シフト勤務', '週4日勤務', '休日休暇(金曜日)', 'オフィスが禁煙・分煙']


def one_hot(test):
    code = ["会社概要　業界コード","勤務地　市区町村コード","職種コード"]
    for c in code:
        dic = {v:k for k,v in enumerate(test[c].value_counts().keys())}
        test[c] = test[c].map(lambda x : dic[x])
    return test
def get_data(filename):
    test = pd.read_csv(f"./upload/{filename}")
    no = test["お仕事No."]
    del test["お仕事No."]
    #test  = one_hot(test[col])
    return test[col].values,no
