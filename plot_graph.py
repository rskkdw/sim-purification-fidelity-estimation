import csv, sys
from collections import defaultdict


def merge_data(stats_file, direc):
    # 入力ファイル名と出力ファイル名
    merged_file = direc+'/merged.csv'

    # 結果を保持する辞書
    merged_data = defaultdict(lambda: {
        "shots": 0,
        "errors": 0,
        "discards": 0,
        "seconds": 0.0,
        "decoder": "",
        "strong_id": "",
        "json_metadata": "",
        "custom_counts": ""
    })

    # CSVファイルを読み込み、データをマージ
    with open(stats_file, 'r') as csvfile:
        reader = csv.DictReader(csvfile)
        reader.fieldnames = [field.strip() for field in reader.fieldnames]

        for row in reader:
            row = {key.strip(): value.strip() for key, value in row.items()}

            metadata_dict = json.loads(row['json_metadata'])
            custom_counts = json.loads(row['custom_counts'])
            
            # 文字列キーをtupleなどのハッシュ可能型にする
            key = (metadata_dict['depth'], metadata_dict['errortype'], metadata_dict['p'], metadata_dict['post_selection_basis'], metadata_dict['bell_pair_infidelity'])

            if key not in merged_data:
                merged_data[key] = {
                    'decoder': row['decoder'],
                    'strong_id': row['strong_id'],
                    'json_metadata': metadata_dict,
                    'custom_counts': custom_counts,
                    'shots': int(row['shots']),
                    'errors': int(row['errors']),
                    'discards': int(row['discards']),
                    'seconds': float(row['seconds']),
                }
            else:
                merged_data[key]['shots'] += int(row['shots'])
                merged_data[key]['errors'] += int(row['errors'])
                merged_data[key]['discards'] += int(row['discards'])
                merged_data[key]['seconds'] += float(row['seconds'])
                merged_data[key]['custom_counts']['detection_events'] += custom_counts['detection_events']
                merged_data[key]['custom_counts']['detectors_checked'] += custom_counts['detectors_checked']
    # マージされたデータをCSVファイルに書き込み
    with open(merged_file, 'w', newline='') as outfile:
        fieldnames = ['shots', 'errors', 'discards', 'seconds', 'decoder', 'strong_id', 'json_metadata', 'custom_counts']
        writer = csv.DictWriter(outfile, fieldnames=fieldnames)
        writer.writeheader()

        for key, data in merged_data.items():
            writer.writerow({
                'shots': data['shots'],
                'errors': data['errors'],
                'discards': data['discards'],
                'json_metadata': data['json_metadata'],
                'decoder': data['decoder'],
                'seconds': data['seconds'],
                'strong_id': data['strong_id'],
                'custom_counts': data['custom_counts']
            })

import csv, ast
import json
import matplotlib.pyplot as plt
import numpy as np

def plot_graph(direc):
    # マージされたデータファイル名
    merged_file = direc+'/merged.csv'
    # データを読み込み
    r_values = []
    p_values = []
    error_rates = []
    error_bars = []
    bell_pair_infidelity_values = []

    code_name = None
    # CSVファイルを適切に読み込む
    with open(merged_file, 'r') as infile:
        reader = csv.DictReader(infile, delimiter=',')
        
        for row in reader:
            # 各フィールドから余分なスペースやタブを削除
            row = {key.strip(): value.strip() for key, value in row.items()}
            row['json_metadata'] = ast.literal_eval(row['json_metadata'])
            row['custom_counts'] = ast.literal_eval(row['custom_counts'])

            local_p = row['json_metadata']['p']
            bell_pair_infidelity = row['json_metadata']['bell_pair_infidelity']
            error_type = row['json_metadata']['errortype']
            post_selection_basis = row['json_metadata']['post_selection_basis']
            depth = row['json_metadata']['depth']

            if post_selection_basis != 'all':
                continue

            # r と p を取得
            json_metadata = row['json_metadata']
            depth = int(json_metadata['depth'])
            local_p = float(json_metadata['p'])
            
            shots = int(row['shots'])
            errors = int(row['errors'])
            discards = int(row['discards'])
            
            # エラーレートの計算
            if shots - discards > 0:
                error_rate = errors / (shots - discards)
                # 標準誤差の計算
                error_bar = np.sqrt((error_rate * (1 - error_rate)) / (shots - discards))
            else:
                print(f"Skipping depth {depth} with p={local_p} and bell_pair_infidelity={bell_pair_infidelity} due to zero valid shots after discards.")
                continue
            
            r_values.append(depth)
            p_values.append(local_p)
            error_rates.append(error_rate)
            error_bars.append(error_bar)
            bell_pair_infidelity_values.append(bell_pair_infidelity)

    # プロット用データを準備
    unique_p_values = sorted(set(p_values))  # pの順にソート
    unique_bell_pair_infidelity_values = sorted(set(bell_pair_infidelity_values))  # bell_pair_infidelityの順にソート

    plt.figure(figsize=(10, 6))

    # 各 p ごとにエラーレートをプロット
    for bell_pair_infidelity in unique_bell_pair_infidelity_values:
        for p in unique_p_values:
            # p に対応する r, error_rate, error_bar のリストを取得
            data_for_p_and_f = [(r, er, eb) for r, er, eb, pp, f in zip(r_values, error_rates, error_bars, p_values, bell_pair_infidelity_values) if pp == p and f == bell_pair_infidelity]
            
            # r の値でソート
            data_for_p_and_f.sort(key=lambda x: x[0])
            
            # ソートされたデータから必要な部分を抽出
            depth_for_p_and_f = [item[0] for item in data_for_p_and_f]
            error_rate_for_p_and_f = [item[1] for item in data_for_p_and_f]
            error_bar_for_p_and_f = [item[2] for item in data_for_p_and_f]
            
            plt.errorbar(depth_for_p_and_f, error_rate_for_p_and_f, yerr=error_bar_for_p_and_f, marker='o', label=f'p={p}, bell_pair_infidelity={bell_pair_infidelity}', capsize=5)

    # 軸ラベルとタイトル
    plt.xlabel('Rounds (r)')
    plt.ylabel('Error Rate (errors / (shots - discards))')
    plt.title(f'Error Rate by Rounds and Probability, row_bell_pair_infidelity={unique_bell_pair_infidelity_values}, {error_type}_error')
    plt.yscale('log')

    # 凡例の表示 (pが小さい方が下に位置するように)
    handles, labels = plt.gca().get_legend_handles_labels()
    plt.legend(list(reversed(handles)), list(reversed(labels)))

    # 画像を保存
    graph_file = direc+f"/{code_name}_{error_type}error.png"
    plt.savefig(graph_file)

    # グラフを表示
    plt.show()

def main():
    direc = sys.argv[1]
    stats_file = sys.argv[2]
    print(direc, stats_file)
    merge_data(stats_file, direc)
    plot_graph(direc)

if __name__ == '__main__':
    main()
