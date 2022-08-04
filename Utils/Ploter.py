# Designer:Pan YuDong
# Coder:God's hand
# Time:2022/1/28 2:18
import matplotlib
import numpy as np
import matplotlib.pyplot as plt
def plot_save_Result(final_acc_list, model_name, dataset='DatasetA', UD=True, ratio=1, win_size='1', text=True):
    '''
    :param final_acc_list: The final accuracy list
    :param model_name: The name of model to be validated
    :param dataset: The name of dataset to be validated
    :param UD: True——User-Dependent；False——User-Independent
    :param ratio: 1——80% vs 20%;2——50% vs 50%;3——20% vs 80%(UD Approach)
                  0 or else——(N-1)/N vs 1/N(UI Approach)
    :param win_size: The window size used in data to evaluate model performance
    :param text: Is the specific value displayed on the histogram
    :return:
           Show the result figure and save it
    '''
    if ratio == 1:
        proportion = '8vs2'
    elif ratio == 2:
        proportion = '5vs5'
    elif ratio == 3:
        proportion = '2vs8'
    else:
        proportion = 'N-1vs1'

    if UD == True:
        val_way = 'PerSubject'
    else:
        val_way = 'CrossSubject'

    final_acc_list = np.asarray(final_acc_list)
    final_mean_list = np.mean(final_acc_list, axis=0)
    final_var_list = np.std(final_acc_list, ddof=1, axis=0)  # ddof: default——divide N(biased);1——divide N-1(unbiased)
    final_mean_list = np.append(final_mean_list, np.mean(final_mean_list, axis=0))
    final_var_list = np.append(final_var_list, np.std(final_mean_list, ddof=1, axis=0))

    with open(f'../Result/{dataset}/{model_name}/{proportion}_{val_way}_Classification_Result({win_size}S).txt',
              'w') as fw:
        fw.write("final_mean_list:" + str(final_mean_list) + '\r\n')
        fw.write("final_var:" + str(final_var_list) + '\r\n')
        fw.write(f"final_acc_mean = {final_mean_list[-1] * 100:.2f} ± {final_var_list[-1] * 100:.2f}")

    data1 = final_mean_list

    # cal the length of data
    len_data = len(data1)

    # set x axis element for bar
    a = [i for i in range(len_data)]

    # set size of figure
    plt.figure(figsize=(20, 8), dpi=80)

    # adjust the direction of ticks
    matplotlib.rcParams['ytick.direction'] = 'in'

    # add grid for y axis
    # plt.rc('axes', axisbelow=True)  # 将网格始终置于底部
    # plt.grid(axis='y', linestyle='--', linewidth=1, alpha=0.4)

    # plot the bars
    for i in range(len_data):
        plt.bar(a[i], data1[i], width=0.35, label='final')

    # set ticks for x axis and y axis
    x_ticks_bound = [i for i in range(len_data)]
    x_ticks_content = [str(i + 1) for i in range(len_data - 1)]
    x_ticks_content.append('mean')
    plt.xticks(x_ticks_bound, x_ticks_content, fontsize=15)

    y_ticks_bound = [i*0.1 for i in range(11)]
    y_ticks_content = [str(i * 10) for i in range(11)]
    plt.yticks(y_ticks_bound, y_ticks_content, fontsize=15)

    # set label for data
    plt.xlabel('Subject', fontsize=15)
    plt.ylabel('Accuracy(%)', fontsize=15)
    plt.title(f'{proportion} {val_way} Classification Result({win_size}S)', fontsize=15)

    # adjust size of axis
    ax = plt.gca()
    ax.spines['bottom'].set_linewidth(2)
    ax.spines['left'].set_linewidth(2)

    if text == True:
        # text elements on the top of bar
        for i in range(len_data):
           delta = 0.05
           if i != len_data - 1:
               plt.text(a[i] - delta * 3.0, data1[i] + delta * 0.1, f'{data1[i] * 100:.2f}')
           else:
               plt.text(a[i] - delta * 5.0, data1[i] + delta * 0.1,
                        f'{final_mean_list[-1] * 100:.2f}±{final_var_list[-1] * 100:.2f}', color='r')
    plt.savefig(f'../Result/{dataset}/{model_name}/{proportion}_{val_way}_Classification_Result({win_size}S).png')
    plt.show()