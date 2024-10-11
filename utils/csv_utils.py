import pandas as pd
import os

def write_results(results:dict, cur_class, total_classes, csv_path):
    keys = list(results.keys())

    if not os.path.exists(csv_path):
        df_all = None
        for class_name in total_classes:
            r = dict()
            for k in keys:
                r[k] = 0.00
            df_temp = pd.DataFrame(r, index=[class_name])

            if df_all is None:
                df_all = df_temp
            else:
                df_all = pd.concat([df_all, df_temp], axis=0)

        df_all.to_csv(csv_path, header=True, float_format='%.2f')

    df = pd.read_csv(csv_path, index_col=0)

    for k in keys:
        df.loc[cur_class, k] = results[k]

    df.to_csv(csv_path, header=True, float_format='%.2f')

def save_metric_list(metrics_list, total_classes, class_name, dataset, csv_path):
    results = dict()
    for indx, metrics in enumerate(metrics_list):
        results[f'i_roc_{indx}'] = metrics['i_roc']
        results[f'p_roc_{indx}'] = metrics['p_roc']
        results[f'p_pro_{indx}'] = metrics['p_pro']

    if dataset != 'mvtec':
        for indx in range(len(total_classes)):
            total_classes[indx] = f"{dataset}-{total_classes[indx]}"
        class_name = f"{dataset}-{class_name}"
    write_results(results, class_name, total_classes, csv_path)
