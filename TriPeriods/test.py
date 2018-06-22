import numpy as np
import pandas as pd


def ret2tick(ret):
    return ret


def gauss_wave_predict(wave, period, n_fft, n_predict, gauss_alpha):
    """
    高斯滤波提取特定周期成分，通过前向补零提升分辨率
    :param wave:        输入序列，为列向量
    :param period:      需要提取的周期长度，单位为月
    :param n_fft:       FFT长度，也即填0后的长度
    :param n_predict:   外延预测的长度
    :param gauss_alpha: 高斯滤波器带宽
    :return:            滤波提取的目标周期成分，长度为输入长度+n_predict
    """
    # 填充0
    wave_pad = np.pad(wave, (n_fft - len(wave), 0), 'constant') if n_fft > len(wave) else wave

    # 傅里叶变换
    wave_fft = np.fft.fft(wave_pad, n_fft)

    # 生成高斯窗口
    gauss_index = np.arange(1, n_fft + 1)
    center_frequency = n_fft / period + 1
    gauss_win = np.exp(- (gauss_index - center_frequency) ** 2 / gauss_alpha ** 2)

    # 滤波
    wave_filter = wave_fft * gauss_win
    if n_fft % 2 == 0:
        wave_filter[int(n_fft / 2 + 1):] = wave_filter[int(n_fft / 2 - 1):0:-1].conjugate()
    else:
        wave_filter[int(n_fft / 2 + 1):] = wave_filter[int(n_fft / 2):0:-1].conjugate()

    # 逆变换还原序列
    ret = np.fft.ifft(wave_filter).real
    output = np.append(ret[-len(wave):], ret[:n_predict])
    return output


def cal_indus_order(yoy_wave, yoy_base, periods, window_size, fft_size, gauss_alpha):
    """
    利用周期建模得到对下一期行业收益率的预测排名
    :param yoy_wave:    行索引为日期，列索引为行业，值为同比序列
    :param yoy_base:    列向量，基准的同比序列
    :param periods:     三周期长度
    :param window_size: 窗口长度，在该窗口内进行周期三因子滤波拟合，并外延
    :param fft_size:    傅里叶变换长度，这里长于原始序列，为了提升频域分辨率
    :param gauss_alpha: 高斯滤波器带宽
    :return:
    """
    # 获取截面个数，行业个数
    panel_num, indus_num = yoy_wave.shape

    # 遍历每个截面，通过周期建模，获取每个行业变化幅度
    change = np.zeros((panel_num - window_size + 1, indus_num))
    for p in range(window_size, panel_num + 1):
        # 基准同比序列历史序列
        base_seq = yoy_base[p - window_size:p]
        # 抽取基准三周期滤波结果作为统一定价因子
        base_filter = np.zeros((len(base_seq) + 1, len(periods)))
        for i in range(len(periods)):
            base_filter[:, i] = gauss_wave_predict(base_seq, periods[i], fft_size, 1, gauss_alpha)

        # 遍历行业，对定价因子做回归
        for i in range(indus_num):
            # 获取训练窗口期内的同比序列
            y = yoy_wave[p - window_size:p, i]
            y_len = len(y)
            # 样本内回归，获取回归系数
            x = np.column_stack((base_filter[:y_len, ], np.ones(y_len)))
            b = np.linalg.lstsq(x, y)[0]
            # 拟合值和预测值
            cur = np.dot(np.append(np.asarray(base_filter[y_len-1, ]), 1), b)
            predict = np.dot(np.append(np.asarray(base_filter[-1, ]), 1), b)
            change[p - window_size, i] = predict - cur

    indus_order = np.zeros_like(change)
    for p in range(change.shape[0]):
        """
        存疑，为什么原始代码里sort了两次
        """
        indus_order[p, ] = np.argsort(change[p, ])[::-1]
        indus_order[p, ] = np.argsort(indus_order[p, ])

    return indus_order


def cal_long_weight(indus_order, long_weight):
    """
    计算截面持仓权重，针对多头回测场景
    :param indus_order: 行业排名预测情况，行索引为截面，列索引为行业
    :param long_weight: 截面配置明细，输入为向量，长度代表配置的行业数目
    :return:            权重矩阵，行索引为行业，列索引为时间
    """
    # 归一化
    long_weight = np.asarray(long_weight) / sum(long_weight)

    # 初始化结果
    w = np.zeros_like(indus_order)

    # 排名替换
    for i in range(len(long_weight)):
        w[indus_order == i] = long_weight(i)

    return w.T


def cal_nav(w, bktest_param, global_param):
    """
    计算组合的净值走势
    :param w:               每个截面各组合的权重
    :param bktest_param:    回测参数
    :param global_param:    全局参数
    :return:                各组合净值走势
    """
    # 获取参数，初始化结果
    daily_dates = global_param.daily_dates          # 全日期序列
    close = global_param.daily_close                # 全日期对应的行业收盘点位
    start_loc = bktest_param.start_loc              # 回测起始日期在全日期序列中的位置
    end_loc = bktest_param.end_loc                  # 回测结束日期在全日期序列中的位置
    commission_rate = bktest_param.commission_rate  # 交易费率
    refresh_dates = bktest_param.refresh_dates      # 回测区间内的调仓日期序列
    predict_dates = bktest_param.predict_dates      # 所有有效预测截面日期
    nav = np.zeros((end_loc - start_loc + 1))       # 初始化结果

    # 起始日就是第一个换仓日，这里构建初始仓位
    refresh_index = 1
    cur_date_index = start_loc  # 与全日期序列对齐
    prev_date_index = cur_date_index - 1  # 与全日期序列对齐
    prev_date = daily_dates[prev_date_index]
    prev_panel_index = list(daily_dates).index(prev_date)  # 有效预测序列 yoy_dates 对齐, 便与w对齐
    refresh_w = w[:, prev_panel_index]

    # 建仓完毕，扣除手续费
    nav[cur_date_index - start_loc] = 1 - commission_rate
    last_portfolio = (1 - commission_rate) * refresh_w

    # 按日期频率遍历，更新净值，遇到调仓日更换仓位
    for d in range(start_loc + 1, end_loc + 1):
        # 执行净值更新，分空仓和不空仓情形
        if sum(last_portfolio) == 0:
            nav[d - start_loc + 1] = nav[d - start_loc]
        else:
            last_portfolio = close[d, ] / close[d - 1, ] * last_portfolio
            nav[d - start_loc + 1] = np.nansum(last_portfolio)

        # 判断当前日期是否为新的调仓日，是则进行调仓
        if refresh_index < len(refresh_dates) - 1 and daily_dates[d] == refresh_dates[refresh_index + 1]:
            # 将最新仓位归一化
            last_w = np.zeros_like(last_portfolio) if sum(last_portfolio) == 0 \
                else last_portfolio / sum(last_portfolio)

            # 获取最新的权重分布，注意这里的权重分布是归一化的结果
            refresh_index += 1
            cur_date_index = d
            prev_date_index = cur_date_index - 1
            prev_date = daily_dates[prev_date_index]
            prev_panel_index = list(predict_dates).index(prev_date)
            refresh_w = w[:, prev_panel_index]

            # 根据前后权重差别计算换手率，调整净值
            refresh_turn = sum(abs(refresh_w - last_w))
            nav[d - start_loc + 1] = nav[d - start_loc + 1] * (1 - refresh_turn * commission_rate)
            last_portfolio = nav[d - start_loc + 1] * refresh_w


def performance(nav, benchmark, refresh_index):
    """
    计算组合净值的风险收益指标
    :param nav:             组合净值矩阵，其中每列代表一个组合
    :param benchmark:       基准净值
    :param refresh_index:   调仓索引，用于计算调仓的胜率
    :return:                每行代表一个组合，每列代表不同的指标
    """
    # 获取净值矩阵大小
    nav = nav if len(nav.shape) == 2 else nav.reshape(len(nav), 1)
    m, n = nav.shape

    # 初始化结果矩阵
    perf = pd.DataFrame(np.zeros(n + 1, 9))
    perf.columns = ['年化收益率', '年化波动率', '夏普比率', '最大回撤',
                    '年化超额收益率', '超额收益年化波动率', '信息比率',
                    '相对基准月胜率', '超额收益最大回撤']

    # 计算每个组合的相关指标
    for i in range(n):
        # 年化收益率
        perf.iloc[i, 1] = nav[-1, i] ** (250 / m) - 1
        # 年化波动率
        perf.iloc[i, 2] = np.std(nav[1:, i]) / nav[:, i] * 250 ** 0.5
        # 夏普比率
        perf.iloc[i, 3] = perf.iloc[i, 1] / perf.iloc[i, 2]
        # 最大回撤
        max_draw_down = 0
        for d in range(m):
            cur_draw_down = nav[d, i] / np.max(nav[:, i]) - 1
            if cur_draw_down < max_draw_down:
                max_draw_down = cur_draw_down

        perf.iloc[i, 4] = max_draw_down
        # 年化超额收益率
        rp = nav[1:, i] / nav[:-1, i]
        rb = benchmark[1:, i] / nav[:-1, i]
        excess = rp - rb
        cum_excess = ret2tick(excess)
        perf.iloc[i, 5] = cum_excess[-1] ** (250 / m) - 1
        # 超额收益年化波动率
        perf.iloc[i, 6] = np.std(excess) * 250 ** 0.5
        # 信息比率
        perf.iloc[i, 7] = perf.iloc[i, 5] / perf.iloc[i, 6]
        # 相对基准月胜率


if __name__ == '__main__':
    b = np.arange(4, 29, step=3)
    print(len(b.shape))

    # a1 = range(1, 28, 3)
    # a2 = range(2, 35, 4)
    # a3 = range(3, 44, 5)
    # a = np.asmatrix([a1, a2, a3]).T
    # order = cal_indus_order(a, b, [2, 4, 6], 5, 6, 5)

