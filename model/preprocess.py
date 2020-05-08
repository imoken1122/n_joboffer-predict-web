from model.text_preprocess import _oubo,_haizoku,tfidf_transformer
from model.umap_transfrom import umap_transfromer
import numpy as np
import pandas as pd
import pandas as pd
import numpy as np
import MeCab
import os
import json
import re
from tqdm import tqdm
drop1 = ['Wワーク・副業可能','休日休暇(金曜日)', '大量募集', '短時間勤務OK(1日4h以内)', '（派遣以外）応募後の流れ', 'ネットワーク関連のスキルを活かす', '応募先\u3000所在地\u3000市区町村', 'ブロックコード1', '給与\u3000経験者給与上限', '未使用.22', '未使用.5', '未使用.16', '電話応対なし', '未使用.2', 'WEB関連のスキルを活かす', '応募拠点', '日払い', '経験必須', '未使用.11', '応募先\u3000名称', '応募先\u3000備考', '未使用.4', '勤務地\u3000最寄駅3（沿線名）', '待遇・福利厚生', '主婦(ママ)・主夫歓迎', '未使用.9', '未使用.19', 'これまでの採用者例', 'ブランクOK', 'プログラム関連のスキルを活かす', '給与\u3000経験者給与下限', '仕事写真（下）\u3000写真1\u3000コメント', '週払い', '17時以降出社OK', '勤務地\u3000最寄駅3（分）', '固定残業制 残業代に充当する労働時間数 上限', '未使用.10', 'メモ', 'フリー項目\u3000タイトル', '（派遣先）概要\u3000従業員数', '応募先\u3000最寄駅（駅名）', '固定残業制 残業代 下限', '仕事写真（下）\u3000写真2\u3000ファイル名', '人材紹介', '未使用.6', '（派遣先）勤務先写真コメント', '寮・社宅あり', '外国人活躍中・留学生歓迎', '未使用.3', '募集形態', '未使用.17', '未使用.1', 'ベンチャー企業', '仕事写真（下）\u3000写真2\u3000コメント', '未使用', '未使用.21', 'WEB面接OK', '学生歓迎', '未使用.7', '未使用.14', '先輩からのメッセージ', '未使用.15', 'ブロックコード3', '（派遣先）概要\u3000勤務先名（フリガナ）', '固定残業制 残業代 上限', '応募先\u3000所在地\u3000都道府県', '無期雇用派遣', '応募先\u3000最寄駅（沿線名）', '未使用.12', 'バイク・自転車通勤OK', '仕事写真（下）\u3000写真3\u3000コメント', '勤務地\u3000周辺情報', '未使用.18', '仕事写真（下）\u3000写真1\u3000ファイル名', '未使用.20', '勤務地\u3000最寄駅3（駅名）', 'フリー項目\u3000内容', '仕事写真（下）\u3000写真3\u3000ファイル名', '少人数の職場', '未使用.8', 'オープニングスタッフ', '固定残業制 残業代に充当する労働時間数 下限', 'エルダー（50〜）活躍中', 'ブロックコード2', '勤務地\u3000最寄駅3（駅からの交通手段）', '未使用.13', 'シニア（60〜）歓迎', '応募先\u3000所在地\u3000ブロックコード']
drop2 = ['17時以降出社OK',
 '20代活躍中',
 '30代活躍中',
 'CAD関連のスキルを活かす',
 'DTP関連のスキルを活かす',
 'Dip JobsリスティングS',
 'PCスキル不要',
 'WEB登録OK',
 'WEB関連のスキルを活かす',
 'WEB面接OK',
 'Wワーク・副業可能',
 'これまでの採用者例',
 'エルダー（50〜）活躍中',
 'オープニングスタッフ',
 'シニア（60〜）歓迎',
 'ネットワーク関連のスキルを活かす',
 'バイク・自転車通勤OK',
 'フリー項目\u3000タイトル',
 'フリー項目\u3000内容',
 'ブランクOK',
 'ブロックコード1',
 'ブロックコード2',
 'ブロックコード3',
 'プログラム関連のスキルを活かす',
 'ベンチャー企業',
 'ミドル（40〜）活躍中',
 'メモ',
 'ルーティンワークがメイン',
 '主婦(ママ)・主夫歓迎',
 '人材紹介',
 '仕事写真（下）\u3000写真1\u3000コメント',
 '仕事写真（下）\u3000写真1\u3000ファイル名',
 '仕事写真（下）\u3000写真2\u3000コメント',
 '仕事写真（下）\u3000写真2\u3000ファイル名',
 '仕事写真（下）\u3000写真3\u3000コメント',
 '仕事写真（下）\u3000写真3\u3000ファイル名',
 '休日休暇(日曜日)',
 '先輩からのメッセージ',
 '公開区分',
 '動画コメント',
 '動画タイトル',
 '募集形態',
 '勤務地\u3000周辺情報',
 '勤務地\u3000最寄駅3（分）',
 '勤務地\u3000最寄駅3（沿線名）',
 '勤務地\u3000最寄駅3（駅からの交通手段）',
 '勤務地\u3000最寄駅3（駅名）',
 '勤務地固定',
 '固定残業制',
 '固定残業制 残業代 上限',
 '固定残業制 残業代 下限',
 '固定残業制 残業代に充当する労働時間数 上限',
 '固定残業制 残業代に充当する労働時間数 下限',
 '土日祝のみ勤務',
 '土日祝休み',
 '外国人活躍中・留学生歓迎',
 '学生歓迎',
 '寮・社宅あり',
 '対象者設定\u3000年齢上限',
 '対象者設定\u3000年齢下限',
 '対象者設定\u3000性別',
 '少人数の職場',
 '就業形態区分',
 '履歴書不要',
 '平日休みあり',
 '待遇・福利厚生',
 '応募先\u3000備考',
 '応募先\u3000名称',
 '応募先\u3000所在地\u3000ブロックコード',
 '応募先\u3000所在地\u3000市区町村',
 '応募先\u3000所在地\u3000都道府県',
 '応募先\u3000最寄駅（沿線名）',
 '応募先\u3000最寄駅（駅名）',
 '応募拠点',
 '扶養控除内',
 '拠点番号',
 '掲載期間\u3000開始日',
 '新卒・第二新卒歓迎',
 '日払い',
 '未使用',
 '未使用.1',
 '未使用.10',
 '未使用.11',
 '未使用.12',
 '未使用.13',
 '未使用.14',
 '未使用.15',
 '未使用.16',
 '未使用.17',
 '未使用.18',
 '未使用.19',
 '未使用.2',
 '未使用.20',
 '未使用.21',
 '未使用.22',
 '未使用.3',
 '未使用.4',
 '未使用.5',
 '未使用.6',
 '未使用.7',
 '未使用.8',
 '未使用.9',
 '検索対象エリア',
 '残業月10時間未満',
 '残業月20時間未満',
 '派遣会社のうれしい特典',
 '無期雇用派遣',
 '産休育休取得事例あり',
 '研修制度あり',
 '社会保険制度あり',
 '紹介予定派遣',
 '経験必須',
 '経験者優遇',
 '給与/交通費\u3000交通費',
 '給与/交通費\u3000給与上限',
 '給与/交通費\u3000給与下限',
 '給与/交通費\u3000給与支払区分',
 '給与\u3000経験者給与上限',
 '給与\u3000経験者給与下限',
 '英語力不要',
 '資格取得支援制度あり',
 '週1日からOK',
 '週2・3日OK',
 '週払い',
 '雇用形態',
 '電話応対なし',
 '（派遣以外）応募後の流れ',
 '（派遣先）勤務先写真コメント',
 '（派遣先）勤務先写真ファイル名',
 '（派遣先）概要\u3000事業内容',
 '（派遣先）概要\u3000勤務先名（フリガナ）',
 '（派遣先）概要\u3000勤務先名（漢字）',
 '（派遣先）概要\u3000従業員数',
 '（派遣先）配属先部署\u3000男女比\u3000男',
 '（派遣）応募後の流れ',
 '（紹介予定）休日休暇',
 '（紹介予定）入社後の雇用形態',
 '（紹介予定）入社時期',
 '（紹介予定）年収・給与例',
 '（紹介予定）待遇・福利厚生',
 '（紹介予定）雇用形態備考']
pol_col = ['16時前退社OK_10時以降出社OK_XOR',
 '1日7時間以下勤務OK_10時以降出社OK_XOR',
 '1日7時間以下勤務OK_16時前退社OK_XOR',
 'Excelのスキルを活かす_10時以降出社OK_XOR',
 'Excelのスキルを活かす_16時前退社OK_XOR',
 'PowerPointのスキルを活かす_10時以降出社OK_XOR',
 'PowerPointのスキルを活かす_16時前退社OK_XOR',
 'PowerPointのスキルを活かす_1日7時間以下勤務OK_XOR',
 'PowerPointのスキルを活かす_Excelのスキルを活かす_XOR',
 'オフィスが禁煙・分煙_Excelのスキルを活かす_XOR',
 'シフト勤務_10時以降出社OK_XOR',
 'シフト勤務_16時前退社OK_XOR',
 'シフト勤務_Excelのスキルを活かす_XOR',
 '交通費別途支給_10時以降出社OK_XOR',
 '交通費別途支給_16時前退社OK_XOR',
 '交通費別途支給_1日7時間以下勤務OK_XOR',
 '交通費別途支給_Excelのスキルを活かす_XOR',
 '交通費別途支給_PowerPointのスキルを活かす_XOR',
 '休日休暇(土曜日)_Excelのスキルを活かす_XOR',
 '休日休暇(土曜日)_シフト勤務_XOR',
 '休日休暇(木曜日)_16時前退社OK_XOR',
 '休日休暇(木曜日)_Excelのスキルを活かす_XOR',
 '休日休暇(木曜日)_オフィスが禁煙・分煙_XOR',
 '休日休暇(木曜日)_シフト勤務_XOR',
 '休日休暇(火曜日)_Excelのスキルを活かす_XOR',
 '休日休暇(火曜日)_休日休暇(木曜日)_XOR',
 '休日休暇(祝日)_Excelのスキルを活かす_XOR',
 '休日休暇(祝日)_シフト勤務_XOR',
 '休日休暇(祝日)_休日休暇(土曜日)_XOR',
 '休日休暇(祝日)_休日休暇(木曜日)_XOR',
 '制服あり_10時以降出社OK_XOR',
 '制服あり_16時前退社OK_XOR',
 '制服あり_1日7時間以下勤務OK_XOR',
 '制服あり_Excelのスキルを活かす_XOR',
 '制服あり_PowerPointのスキルを活かす_XOR',
 '制服あり_交通費別途支給_XOR',
 '外資系企業_16時前退社OK_XOR',
 '外資系企業_Excelのスキルを活かす_XOR',
 '外資系企業_オフィスが禁煙・分煙_XOR',
 '外資系企業_シフト勤務_XOR',
 '外資系企業_休日休暇(木曜日)_XOR',
 '外資系企業_休日休暇(祝日)_XOR',
 '大手企業_10時以降出社OK_XOR',
 '大手企業_16時前退社OK_XOR',
 '大手企業_1日7時間以下勤務OK_XOR',
 '大手企業_PowerPointのスキルを活かす_XOR',
 '大手企業_交通費別途支給_XOR',
 '大手企業_制服あり_XOR',
 '学校・公的機関（官公庁）_10時以降出社OK_XOR',
 '学校・公的機関（官公庁）_16時前退社OK_XOR',
 '学校・公的機関（官公庁）_1日7時間以下勤務OK_XOR',
 '学校・公的機関（官公庁）_Excelのスキルを活かす_XOR',
 '学校・公的機関（官公庁）_PowerPointのスキルを活かす_XOR',
 '学校・公的機関（官公庁）_シフト勤務_XOR',
 '学校・公的機関（官公庁）_交通費別途支給_XOR',
 '学校・公的機関（官公庁）_休日休暇(木曜日)_XOR',
 '学校・公的機関（官公庁）_制服あり_XOR',
 '学校・公的機関（官公庁）_外資系企業_XOR',
 '学校・公的機関（官公庁）_大手企業_XOR',
 '服装自由_16時前退社OK_XOR',
 '服装自由_Excelのスキルを活かす_XOR',
 '服装自由_制服あり_XOR',
 '服装自由_学校・公的機関（官公庁）_XOR',
 '未経験OK_10時以降出社OK_XOR',
 '未経験OK_16時前退社OK_XOR',
 '未経験OK_1日7時間以下勤務OK_XOR',
 '未経験OK_Excelのスキルを活かす_XOR',
 '未経験OK_PowerPointのスキルを活かす_XOR',
 '未経験OK_シフト勤務_XOR',
 '未経験OK_交通費別途支給_XOR',
 '未経験OK_制服あり_XOR',
 '未経験OK_大手企業_XOR',
 '未経験OK_学校・公的機関（官公庁）_XOR',
 '正社員登用あり_10時以降出社OK_XOR',
 '正社員登用あり_16時前退社OK_XOR',
 '正社員登用あり_1日7時間以下勤務OK_XOR',
 '正社員登用あり_Excelのスキルを活かす_XOR',
 '正社員登用あり_PowerPointのスキルを活かす_XOR',
 '正社員登用あり_シフト勤務_XOR',
 '正社員登用あり_交通費別途支給_XOR',
 '正社員登用あり_制服あり_XOR',
 '正社員登用あり_大手企業_XOR',
 '正社員登用あり_学校・公的機関（官公庁）_XOR',
 '正社員登用あり_服装自由_XOR',
 '正社員登用あり_未経験OK_XOR',
 '残業なし_10時以降出社OK_XOR',
 '残業なし_16時前退社OK_XOR',
 '残業なし_1日7時間以下勤務OK_XOR',
 '残業なし_PowerPointのスキルを活かす_XOR',
 '残業なし_交通費別途支給_XOR',
 '残業なし_制服あり_XOR',
 '残業なし_大手企業_XOR',
 '残業なし_学校・公的機関（官公庁）_XOR',
 '残業なし_未経験OK_XOR',
 '残業なし_正社員登用あり_XOR',
 '残業月20時間以上_10時以降出社OK_XOR',
 '残業月20時間以上_16時前退社OK_XOR',
 '残業月20時間以上_Excelのスキルを活かす_XOR',
 '残業月20時間以上_PowerPointのスキルを活かす_XOR',
 '残業月20時間以上_シフト勤務_XOR',
 '残業月20時間以上_休日休暇(木曜日)_XOR',
 '残業月20時間以上_制服あり_XOR',
 '残業月20時間以上_外資系企業_XOR',
 '残業月20時間以上_学校・公的機関（官公庁）_XOR',
 '残業月20時間以上_未経験OK_XOR',
 '残業月20時間以上_正社員登用あり_XOR',
 '派遣スタッフ活躍中_10時以降出社OK_XOR',
 '派遣スタッフ活躍中_16時前退社OK_XOR',
 '派遣スタッフ活躍中_1日7時間以下勤務OK_XOR',
 '派遣スタッフ活躍中_PowerPointのスキルを活かす_XOR',
 '派遣スタッフ活躍中_交通費別途支給_XOR',
 '派遣スタッフ活躍中_制服あり_XOR',
 '派遣スタッフ活躍中_大手企業_XOR',
 '派遣スタッフ活躍中_学校・公的機関（官公庁）_XOR',
 '派遣スタッフ活躍中_未経験OK_XOR',
 '派遣スタッフ活躍中_正社員登用あり_XOR',
 '派遣スタッフ活躍中_残業なし_XOR',
 '派遣形態_Excelのスキルを活かす_XOR',
 '派遣形態_正社員登用あり_XOR',
 '社員食堂あり_10時以降出社OK_XOR',
 '社員食堂あり_16時前退社OK_XOR',
 '社員食堂あり_1日7時間以下勤務OK_XOR',
 '社員食堂あり_Excelのスキルを活かす_XOR',
 '社員食堂あり_PowerPointのスキルを活かす_XOR',
 '社員食堂あり_交通費別途支給_XOR',
 '社員食堂あり_制服あり_XOR',
 '社員食堂あり_大手企業_XOR',
 '社員食堂あり_学校・公的機関（官公庁）_XOR',
 '社員食堂あり_未経験OK_XOR',
 '社員食堂あり_正社員登用あり_XOR',
 '社員食堂あり_残業なし_XOR',
 '社員食堂あり_残業月20時間以上_XOR',
 '社員食堂あり_派遣スタッフ活躍中_XOR',
 '英語力を活かす_10時以降出社OK_XOR',
 '英語力を活かす_16時前退社OK_XOR',
 '英語力を活かす_Excelのスキルを活かす_XOR',
 '英語力を活かす_PowerPointのスキルを活かす_XOR',
 '英語力を活かす_シフト勤務_XOR',
 '英語力を活かす_交通費別途支給_XOR',
 '英語力を活かす_休日休暇(木曜日)_XOR',
 '英語力を活かす_制服あり_XOR',
 '英語力を活かす_外資系企業_XOR',
 '英語力を活かす_大手企業_XOR',
 '英語力を活かす_学校・公的機関（官公庁）_XOR',
 '英語力を活かす_未経験OK_XOR',
 '英語力を活かす_正社員登用あり_XOR',
 '英語力を活かす_残業なし_XOR',
 '英語力を活かす_残業月20時間以上_XOR',
 '英語力を活かす_社員食堂あり_XOR',
 '車通勤OK_10時以降出社OK_XOR',
 '車通勤OK_16時前退社OK_XOR',
 '車通勤OK_1日7時間以下勤務OK_XOR',
 '車通勤OK_Excelのスキルを活かす_XOR',
 '車通勤OK_PowerPointのスキルを活かす_XOR',
 '車通勤OK_交通費別途支給_XOR',
 '車通勤OK_制服あり_XOR',
 '車通勤OK_大手企業_XOR',
 '車通勤OK_学校・公的機関（官公庁）_XOR',
 '車通勤OK_服装自由_XOR',
 '車通勤OK_未経験OK_XOR',
 '車通勤OK_正社員登用あり_XOR',
 '車通勤OK_残業なし_XOR',
 '車通勤OK_残業月20時間以上_XOR',
 '車通勤OK_派遣スタッフ活躍中_XOR',
 '車通勤OK_社員食堂あり_XOR',
 '車通勤OK_英語力を活かす_XOR',
 '駅から徒歩5分以内_10時以降出社OK_XOR',
 '駅から徒歩5分以内_16時前退社OK_XOR',
 '駅から徒歩5分以内_1日7時間以下勤務OK_XOR',
 '駅から徒歩5分以内_PowerPointのスキルを活かす_XOR',
 '駅から徒歩5分以内_交通費別途支給_XOR',
 '駅から徒歩5分以内_制服あり_XOR',
 '駅から徒歩5分以内_大手企業_XOR',
 '駅から徒歩5分以内_学校・公的機関（官公庁）_XOR',
 '駅から徒歩5分以内_未経験OK_XOR',
 '駅から徒歩5分以内_正社員登用あり_XOR',
 '駅から徒歩5分以内_残業なし_XOR',
 '駅から徒歩5分以内_派遣スタッフ活躍中_XOR',
 '駅から徒歩5分以内_社員食堂あり_XOR',
 '駅から徒歩5分以内_車通勤OK_XOR']
ks_col = ['勤務地\u3000都道府県コード',
 'フラグオプション選択',
 '勤務地\u3000最寄駅2（駅からの交通手段）',
 '勤務地\u3000最寄駅2（分）',
 '勤務地\u3000市区町村コード',
 '車通勤OK',
 '派遣スタッフ活躍中',
 '駅から徒歩5分以内',
 '勤務地\u3000最寄駅1（分）',
 '勤務地\u3000最寄駅1（駅からの交通手段）',
 '服装自由',
 '交通費別途支給',
 '制服あり',
 '未経験OK',
 '（派遣先）配属先部署\u3000平均年齢',
 '派遣形態',
 '10時以降出社OK',
 '会社概要\u3000業界コード',
 '職種コード',
 '正社員登用あり',
 '社員食堂あり',
 '（派遣先）配属先部署\u3000男女比\u3000女',
 '学校・公的機関（官公庁）']





def onehot(test):
    code = ["会社概要　業界コード","勤務地　市区町村コード","職種コード"]
    label = ["gyoukai_code_dict.json","kinmuti_code_dict.json","syokuba_code_dict.json"]
    for c,l in zip(code,label):
        with open(f'./model/json_dict/{l}') as f:
            dic = json.load(f)
        test[c] = test[c].map(lambda x : dic[f"{x}"])

    return test

def kyuuyo(test):
    
    test.fillna("0",inplace =True )
    for i in tqdm(range(len(test))):
        if re.search("】(.*)万",test.iloc[i]) == None:
            test.iloc[i] = np.nan
        else:
            s = "".join(re.findall("】(.*)万",test.iloc[i]))
            test.iloc[i]= float(s)
    return test.astype(float)

def kinm_time_(data):
    kinm_time = pd.DataFrame()
    for i in tqdm(range(len(data))):
        a = data["期間・時間\u3000勤務時間"].iloc[i]
        l = [int(re.sub(":.*","",i)) for i in re.findall("(.*)\u3000",a)[0].split("〜")]
        kinm_time.loc[i,"start"] = l[0]
        kinm_time.loc[i,"finish"] = l[1]
    return kinm_time


def prefecture(clean_data):
    l = []
    tagger = MeCab.Tagger("-Ochasen")
    for text in clean_data:
        l.append(tagger.parse(text).split()[0])
    return l


def interact_encoding(data,f):
    new = pd.DataFrame()
    for i in tqdm(range(len(f))):
        for j in range(i):
            c,cc = f[i],f[j]
            new["add_" + c +"_"+ cc ] = data[c].values + data[cc].values
            new["minus_" + c +"_"+ cc ] = data[c].values- data[cc].values
            new["dot_" + c +"_"+ cc ] = data[c].values*data[cc].values
    return new

def polynominal_encoding(data,f):
    new = pd.DataFrame()
    for i in tqdm(range(len(f))):
        for j in range(i):            
            new[f"{f[i]}_{f[j]}_XOR"] = np.logical_xor(data[f[i]], data[f[j]])  

    return new


def add_ZeroAgg(data):
    flist = [x for x in data.columns if not x in ['ID','target']]
    new = pd.DataFrame()
    new['sum_Zeros'] = (data[flist] == 0).astype(int).sum(axis=1)
    new['mean_Zeros'] = (data[flist] == 0).astype(int).mean(axis=1)
    new['median_Zeros'] = (data[flist] == 0).astype(int).median(axis=1)
    new['max_Zeros'] = (data[flist] == 0).astype(int).max(axis=1)
    new['min_Zeros'] = (data[flist] == 0).astype(int).min(axis=1)
    new['std_Zeros'] = (data[flist] == 0).astype(int).std(axis=1)
    new['var_Zeros'] = (data[flist] == 0).astype(int).var(axis=1)
    new['skew_Zeros'] = (data[flist] == 0).astype(int).skew(axis=1)
    new['kurtosis_Zeros'] = (data[flist] == 0).astype(int).kurtosis(axis=1)
    return new


def add_IncAgg(data):
    flist = [x for x in data.columns if not x in ['ID','target']]
    new = pd.DataFrame()
    new['sum_Inc'] = (data[flist] > 0).astype(int).sum(axis=1)
    new['mean_Inc'] = (data[flist] > 0).astype(int).mean(axis=1)
    new['median_Inc'] = (data[flist] > 0).astype(int).median(axis=1)
    new['std_Inc'] = (data[flist] > 0).astype(int).std(axis=1)
    new['var_Inc'] = (data[flist] > 0).astype(int).var(axis=1)
    new['min_Inc'] = (data[flist] > 0).astype(int).min(axis=1)
    new['max_Inc'] = (data[flist] > 0).astype(int).max(axis=1)
    new['skew_Inc'] = (data[flist] > 0).astype(int).skew(axis=1)
    new['kurtosis_Inc'] = (data[flist] > 0).astype(int).kurtosis(axis=1)
    return new


def add_UniqAgg(data):

    new = pd.DataFrame()
    for i in tqdm(range(len(data))):

        new.loc[i,'sum_Uniq'] = data.iloc[i,:].unique().sum()
        new.loc[i,'mean_Uniq'] = (data.iloc[i,:]).unique().mean()
        new.loc[i,'std_Uniq'] = (data.iloc[i,:]).unique().std()
        new.loc[i,'var_Uniq'] =  (data.iloc[i,:]).unique().var()
        new.loc[i,'min_Uniq'] =  (data.iloc[i,:]).unique().max()
        new.loc[i,'max_Uniq'] =  (data.iloc[i,:]).unique().min()
    return new


def data_stat(data):
    
    new = pd.DataFrame()
    c = data.columns

    new["mean"] = data[c].mean(axis = 1)
    new["var"] = data[c].var(axis = 1)
    new["std"] = data[c].std(axis = 1)
    new["max"] = data[c].max(axis = 1)
    new["min"] = data[c].min(axis = 1)
    new["kurtosis"] = data[c].kurtosis(axis =1)
    new["skew"] = data[c].skew(axis= 1)
    return new 
def df_concat(flist):
    add_columns = flist[0].columns.values.tolist()
    add_df = flist[0].values
    for i in flist[1:]:
        add_columns += i.columns.values.tolist()
        add_df = np.c_[add_df,i.values]
    return pd.DataFrame(add_df,columns=add_columns)


def get_data(filename):
    print(filename)
    test = pd.read_csv(f"./upload/{filename}")
    no = test["お仕事No."]
    drop = np.unique(drop1 + drop2)
    test.drop(drop,axis= 1,inplace = True)
    d = ["お仕事No.","動画ファイル名","掲載期間　終了日","期間･時間　備考"]
    test.drop(d,axis = 1,inplace = True)
    
    return test,no


def prerpocess_run(filename):
    test,no = get_data(filename)

    te = onehot(test)

    te['給与/交通費　備考'] = kyuuyo(test['給与/交通費　備考'].copy())
    te = pd.concat([te,kinm_time_(te.copy())],axis = 1)
    del te["期間・時間\u3000勤務時間"]

    with open('./model/json_dict/moyori_dict.json') as f:
        dic = json.load(f)
    te["勤務地\u3000最寄駅1（駅名）"] = te["勤務地\u3000最寄駅1（駅名）"].map(lambda x : dic[x] if x in dic.keys() else 1000)

    te["勤務地\u3000備考"] = prefecture(te["勤務地\u3000備考"].copy())

    with open('./model/json_dict/kinmuti_dict.json') as f:
        dic = json.load(f)
    te["勤務地\u3000備考"] = te["勤務地\u3000備考"].map(lambda x : dic[x])

    intract_col = ["（派遣先）配属先部署\u3000人数","（派遣先）配属先部署\u3000平均年齢","勤務地\u3000備考",'職種コード','正社員登用あり','派遣形態','未経験OK',
               "給与/交通費\u3000備考","勤務地\u3000最寄駅1（分）","（派遣先）配属先部署　男女比　女","会社概要　業界コード","勤務地　市区町村コード"]
    data= te[te.columns[te.dtypes!=object]]

    FE1 = interact_encoding(data,intract_col)

    flist = ['10時以降出社OK',
            '16時前退社OK',
            '1日7時間以下勤務OK',
            'Excelのスキルを活かす',
            'PowerPointのスキルを活かす',
            'オフィスが禁煙・分煙',
            'シフト勤務',
            '交通費別途支給',
            '休日休暇(土曜日)',
            '休日休暇(木曜日)',
            '休日休暇(火曜日)',
            '休日休暇(祝日)',
            '制服あり',
            '外資系企業',
            '大手企業',
            '学校・公的機関（官公庁）',
            '服装自由',
            '未経験OK',
            '正社員登用あり',
            '残業なし',
            '残業月20時間以上',
            '派遣スタッフ活躍中',
            '派遣形態',
            '社員食堂あり',
            '英語力を活かす',
            '車通勤OK',
            '駅から徒歩5分以内']
    FE2 = polynominal_encoding(data,flist)
    FE2 = FE2[pol_col]
    FE3 = add_ZeroAgg(data)
    FE5 = add_IncAgg(data)
    FE6 = add_UniqAgg(data.copy())
    FE7 = data_stat(data)
    add_int_FE = df_concat([FE1,FE2,FE3,FE5,FE6,FE7])

    #concat_data = df_concat([data.drop(ks_col,axis = 1),add_int_FE]) 
    #return concat_data[sorted(concat_data.columns)].values,no 
    te["（派遣先）配属先部署"].fillna("None", inplace = True)
    test_haizoku = _haizoku(te)
    test_oubo = _oubo(te)

    haizoku_tfidf = tfidf_transformer(test_haizoku,"haizoku")
    oubo_tfidf = tfidf_transformer(test_oubo,"oubo")

    haizoku_umap = umap_transfromer(haizoku_tfidf,"haizoku")
    oubo_umap = umap_transfromer(oubo_tfidf,"oubo")

    concat_data = df_concat([data.drop(ks_col,axis = 1),add_int_FE,haizoku_umap,oubo_umap])
    return concat_data[sorted(concat_data.columns)].values,no

