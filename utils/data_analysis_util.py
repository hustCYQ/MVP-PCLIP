import matplotlib.pyplot as plt
import pandas as pd
import os
import os

import matplotlib.pyplot as plt
import pandas as pd


def read_csv(path) -> pd.DataFrame:
    df = pd.read_csv(path, index_col=0)
    return df


def read_all_csv_for_single_backbone(prefix, backbone, view_list, root):
    n_view_df_list = []
    for n_view in view_list:
        vidf_2d_csv_path = os.path.join(root, f'{prefix}_{backbone}_{n_view}.csv')
        vidf_2d_3d_csv_path = os.path.join(root, f'{prefix}_{backbone}_{n_view}_fpfh.csv')
        vidf_3d_csv_path = os.path.join(root, f'{prefix}_{backbone}_0_fpfh.csv')

        vidf_2d_df = read_csv(vidf_2d_csv_path)
        vidf_2d_3d_df = read_csv(vidf_2d_3d_csv_path)
        vidf_3d_df = read_csv(vidf_3d_csv_path)

        # calculate the average
        vidf_2d_average_df = vidf_2d_df.mean()
        vidf_2d_3d_average_df = vidf_2d_3d_df.mean()
        vidf_3d_average_df = vidf_3d_df.mean()

        # concat three df
        prefix_list = ['2D', '2D+3D', '3D']
        df_list = [vidf_2d_average_df, vidf_2d_3d_average_df, vidf_3d_average_df]

        result_dict = {}
        for df, p in zip(df_list, prefix_list):
            for k in df.keys():
                result_dict[f'{p}-{k}'] = df[k]

        single_view_df = pd.DataFrame(data=result_dict, index=[n_view])
        n_view_df_list.append(single_view_df)

    all_view_df = pd.concat(n_view_df_list, axis=0)

    return all_view_df


def plot_single_backbone_result(df: pd.DataFrame, save_path=None):
    legend_map = {
        '2D-i_roc': 'I-ROC-2D',
        '2D-p_pro': 'P-PRO-2D',

        '2D+3D-i_roc': 'I-ROC-2D+3D',
        '2D+3D-p_pro': 'P-PRO-2D+3D',

        '3D-i_roc': 'I-ROC-3D',
        '3D-p_pro': 'P-PRO-3D',
    }

    color_map = {
        '2D-i_roc': 'r',
        '2D-p_pro': 'r',

        '2D+3D-i_roc': 'g',
        '2D+3D-p_pro': 'g',

        '3D-i_roc': 'b',
        '3D-p_pro': 'b',
    }

    linestyle_map = {
        '2D-i_roc': '-',
        '2D-p_pro': '--',

        '2D+3D-i_roc': '-',
        '2D+3D-p_pro': '--',

        '3D-i_roc': '-',
        '3D-p_pro': '--',
    }

    marker_map = {
        '2D-i_roc': "o",
        '2D-p_pro': "s",

        '2D+3D-i_roc': "o",
        '2D+3D-p_pro': "s",

        '3D-i_roc': "o",
        '3D-p_pro': "s"
    }

    # width: 7.16

    full_width = 7.16
    markersize = 2
    figsize = (full_width/6, 1.5)
    fontsize = 7

    fig = plt.figure(figsize=figsize, dpi=600)
    ax1 = fig.add_subplot()
    ax1.tick_params(labelsize=fontsize)

    with plt.style.context(['science', 'ieee', 'no-latex']):
        # plot I-AUROC in the first axes
        legend_list = []
        for k in df.keys():
            if k.find('i_roc') != -1:
                ax1.plot(df[k], color=color_map[k], marker=marker_map[k], linestyle=linestyle_map[k],
                         markersize=markersize)
                legend_list.append(legend_map[k])
        # ax1.legend(legend_list, ncol=len(legend_list), loc='lower right', bbox_to_anchor=[0, 1], fontsize=fontsize)
        ax1.set_ylabel("Image AU-ROC(%)", fontsize=fontsize)
        ax1.set_ylim([77, 96])
        ax2 = ax1.twinx()
        ax2.tick_params(labelsize=fontsize)
        legend_list = []
        for k in df.keys():
            if k.find('p_pro') != -1:
                ax2.plot(df[k], color=color_map[k], marker=marker_map[k], linestyle=linestyle_map[k],
                         markersize=markersize)
                legend_list.append(legend_map[k])
        # ax2.legend(legend_list, ncol=len(legend_list), loc='lower left', bbox_to_anchor=[1, 1], fontsize=fontsize)
        ax2.set_ylabel("Pixel AU-PRO(%)", fontsize=fontsize)
        ax2.set_ylim([84, 94])
        ax1.set_xlabel("Number of views", fontsize=fontsize)
        # ax1.grid(which='both')
        # ax2.grid(which='both')

    if save_path is not None:
        plt.savefig(save_path, bbox_inches="tight", format='svg')
    else:
        plt.show(bbox_inches="tight")


if __name__ == '__main__':
    prefix = 'UNIFORM'
    backbone_list = ['resnet18','resnet34','resnet50','wide_resnet50_2','wide_resnet101_2']
    backbone = 'resnet18'
    fig_save_root = './results/analysis-result'
    view_list = [1, 3, 6, 9, 12, 15, 18, 21, 24, 27]
    root = './results/csv'
    for backbone in backbone_list:
        single_backbone_df = read_all_csv_for_single_backbone(prefix, backbone, view_list, root)
        save_path = os.path.join(fig_save_root, f'{backbone}_vs_n_views.svg')
        plot_single_backbone_result(single_backbone_df, save_path=save_path)
        print(f'save {backbone}.svg')

    # single_backbone_df.to_csv("test.csv", header=True)
