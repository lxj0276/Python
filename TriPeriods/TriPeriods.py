import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pylab import mpl
from dateutil.parser import parse
from datetime import timedelta


class BktestParam:
    init_period = ['2005-03-01', '2018-01-31']
    commission_rate = 0
    periods = [21, 42, 100]
    window_size = 50
    fft_size = 4096
    gauss_alpha = 10
    long_weight = [1]

    predict_dates = None
    refresh_dates = None
    start_date = None
    start_loc = None
    end_date = None
    end_loc = None
    back_dates = None
    monthly_refresh_dates = None


class GlobalParam:
    indus_name = ['周期上游', '周期中游', '周期下游', '大金融', '消费', '成长']
    indus_num = 6
    daily_close = None
    daily_dates = None
    base_nav = None
    monthly_indus = None
    monthly_base = None
    monthly_dates = None
    yoy_indus = None
    yoy_base = None
    yoy_dates = None


class BktestResult:
    indus_order = None
    w = None
    nav = None
    nav_perf = None


def date_num(date_str):
    """
    将日期字符串转换为 MatLab 风格的日期
    :param date_str:    日期字符串
    :return:            MatLab 风格的日期数值
    """
    return (parse(date_str) - parse('0001-01-01')).days + 367


def num_date(num):
    """
    将 MatLab风格的日期数转换为字符串
    :param num:         MatLab风格的日期数值
    :return:            日期字符串
    """
    n = int(num)
    return (parse('0001-01-01') + timedelta(days=n-367)).strftime("%Y-%m-%d")


def gen_global_param(data, industry):
    """
    获取全局参数 GlobalParam
    :param data:        各个行业的净值数据
    :param industry:    行业分类
    """
    # 合成大类板块，计算日频收盘价
    small_indus_return = (data['close'] / data['close'].shift(1) - 1).dropna()
    large_indus_return = np.zeros((len(small_indus_return), GlobalParam.indus_num))
    for i in range(GlobalParam.indus_num):
        large_indus = GlobalParam.indus_name[i]
        include_indus = industry['indus_type'] == large_indus
        large_indus_return[:, i] = np.asarray(small_indus_return.loc[:, include_indus].mean(axis=1))

    daily_close = (large_indus_return + 1).cumprod(axis=0)
    GlobalParam.daily_close = np.row_stack((np.ones(GlobalParam.indus_num), daily_close))
    GlobalParam.daily_dates = np.asarray(data['daily_dates']).reshape(-1)

    # 生成日频行业等权净值
    base_return = large_indus_return.mean(axis=1)
    base_nav = (base_return + 1).cumprod()
    GlobalParam.base_nav = np.append(np.ones(1), base_nav)

    # 从日频数据中抽取月频数据，生成对数同比序列，用于周期建模
    # 月频收盘数据
    GlobalParam.monthly_dates = np.asarray(data['monthly_dates']).reshape(-1)
    month_index = [(i in GlobalParam.monthly_dates) for i in GlobalParam.daily_dates]
    GlobalParam.monthly_indus = GlobalParam.daily_close[month_index, :]
    GlobalParam.monthly_base = GlobalParam.base_nav[month_index]

    # 生成对应对数同比序列
    GlobalParam.yoy_indus = np.log(GlobalParam.monthly_indus[12:] / GlobalParam.monthly_indus[:-12])
    GlobalParam.yoy_base = np.log(GlobalParam.monthly_base[12:] / GlobalParam.monthly_base[:-12])
    GlobalParam.yoy_dates = GlobalParam.monthly_dates[12:]


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
            b = np.linalg.lstsq(x, y, rcond=None)[0]
            # 拟合值和预测值
            cur = np.dot(np.append(np.asarray(base_filter[y_len - 1, ]), 1), b)
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
        w[indus_order == i] = long_weight[i]

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
    cur_date_index = start_loc                                  # 与全日期序列对齐
    prev_date_index = cur_date_index - 1                        # 与全日期序列对齐
    prev_date = daily_dates[prev_date_index]
    prev_panel_index = list(predict_dates).index(prev_date)     # 有效预测序列 yoy_dates 对齐, 便与w对齐
    refresh_w = w[:, prev_panel_index]

    # 建仓完毕，扣除手续费
    nav[cur_date_index - start_loc] = 1 - commission_rate
    last_portfolio = (1 - commission_rate) * refresh_w

    # 按日期频率遍历，更新净值，遇到调仓日更换仓位
    for d in range(start_loc + 1, end_loc + 1):
        # 执行净值更新，分空仓和不空仓情形
        if sum(last_portfolio) == 0:
            nav[d - start_loc] = nav[d - start_loc]
        else:
            last_portfolio = close[d, ] / close[d - 1, ] * last_portfolio
            nav[d - start_loc] = np.nansum(last_portfolio)

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
            nav[d - start_loc + 1] = nav[d - start_loc] * (1 - refresh_turn * commission_rate)
            last_portfolio = nav[d - start_loc] * refresh_w

    return nav


def performance(nav, benchmark, refresh_index):
    """
    计算组合净值的风险收益指标
    :param nav:             组合净值矩阵，其中每列代表一个组合
    :param benchmark:       基准净值
    :param refresh_index:   调仓索引，用于计算调仓的胜率
    :return:                每行代表一个组合，每列代表不同的指标
    """
    # 获取净值矩阵大小
    nav, benchmark, refresh_index = np.asarray(nav), np.asarray(benchmark), np.asarray(refresh_index)
    nav = nav if len(nav.shape) == 2 else nav.reshape(len(nav), 1)
    m, n = nav.shape

    # 初始化结果矩阵
    perf = pd.DataFrame(np.zeros((n + 1, 9)))
    perf.columns = ['年化收益率', '年化波动率', '夏普比率', '最大回撤',
                    '年化超额收益率', '超额收益年化波动率', '信息比率',
                    '相对基准月胜率', '超额收益最大回撤']

    # 计算每个组合的相关指标
    for i in range(n):
        # 年化收益率
        perf.iloc[i, 0] = (nav[-1, i]) ** (250 / m) - 1
        # 年化波动率
        perf.iloc[i, 1] = np.std(nav[1:, i] / nav[:-1, i], ddof=1) * 250 ** 0.5
        # 夏普比率
        perf.iloc[i, 2] = perf.iloc[i, 0] / perf.iloc[i, 1]
        # 最大回撤
        max_draw_down = 0
        for d in range(m):
            cur_draw_down = nav[d, i] / np.max(nav[:d + 1, i]) - 1
            if cur_draw_down < max_draw_down:
                max_draw_down = cur_draw_down

        perf.iloc[i, 3] = max_draw_down
        # 年化超额收益率
        rp = nav[1:, i] / nav[:-1, i]
        rb = benchmark[1:] / benchmark[:-1]
        excess = rp - rb
        cum_excess = np.append(np.arange(1, 2), np.cumprod(1 + excess))
        perf.iloc[i, 4] = (cum_excess[-1]) ** (250 / m) - 1
        # 超额收益年化波动率
        perf.iloc[i, 5] = np.std(excess, ddof=1) * 250 ** 0.5
        # 信息比率
        perf.iloc[i, 6] = perf.iloc[i, 4] / perf.iloc[i, 5]
        # 相对基准月胜率
        close_portfolio = nav[refresh_index, i]
        close_portfolio = np.append(close_portfolio, nav[-1, i])
        portfolio_return = close_portfolio[1:] / close_portfolio[:-1]
        close_benchmark = benchmark[refresh_index]
        close_benchmark = np.append(close_benchmark, benchmark[-1])
        benchmark_return = close_benchmark[1:] / close_benchmark[:-1]

        def win_rate(a1, a2):
            c = 0
            for j in range(len(a1)):
                if a1[j] > a2[j]:
                    c += 1
            return c / len(a1)

        perf.iloc[i, 7] = win_rate(portfolio_return, benchmark_return)

        # 超额收益最大回撤
        max_draw_down = 0

        for d in range(m - 1):
            cur_draw_down = cum_excess[d] / np.max(cum_excess[:d + 1]) - 1
            if cur_draw_down < max_draw_down:
                max_draw_down = cur_draw_down

        perf.iloc[i, 8] = max_draw_down

    # 计算基准各个指标的值
    perf.iloc[-1, 0] = benchmark[-1] ** (250 / m) - 1
    perf.iloc[-1, 1] = np.std(benchmark[1:] / benchmark[:-1], ddof=1) * 250 ** 0.5
    perf.iloc[-1, 2] = perf.iloc[-1, 0] / perf.iloc[-1, 1]

    # 最大回撤
    max_draw_down = 0
    for d in range(m):
        cur_draw_down = benchmark[d] / np.max(benchmark[:d + 1]) - 1
        if cur_draw_down < max_draw_down:
            max_draw_down = cur_draw_down
    perf.iloc[-1, 3] = max_draw_down

    return perf


def bktest_unit():
    """
    回测
    """
    # 有效截面日期序列
    BktestParam.predict_dates = GlobalParam.yoy_dates[BktestParam.window_size-1:]

    # 获取调仓日期序列，按下月月初调仓，注意边界调整
    index = np.arange(len(GlobalParam.daily_dates))
    predict_dates_index = index[[(i in BktestParam.predict_dates) for i in GlobalParam.daily_dates]]
    if predict_dates_index[-1] + 1 <= len(GlobalParam.daily_dates):
        BktestParam.refresh_dates = GlobalParam.daily_dates[predict_dates_index + 1]
    else:
        BktestParam.refresh_dates = GlobalParam.daily_dates[predict_dates_index[:-1] + 1]

    # 暂存初始设置的回测区间
    start_date = BktestParam.init_period[0]
    end_date = BktestParam.init_period[1]

    # 回测开始日期，选择大于等于设置日期的第一个调仓日
    BktestParam.start_date = BktestParam.refresh_dates[BktestParam.refresh_dates >= date_num(start_date)][0]
    BktestParam.start_loc = list(GlobalParam.daily_dates).index(BktestParam.start_date)

    # 回测截止日期，选择该日期最近的一个交易日
    BktestParam.end_date = GlobalParam.daily_dates[GlobalParam.daily_dates <= date_num(end_date)][-1]
    BktestParam.end_loc = list(GlobalParam.daily_dates).index(BktestParam.end_date)

    # 整个回测区间日期序列
    BktestParam.back_dates = GlobalParam.daily_dates[BktestParam.start_loc: BktestParam.end_loc + 1]

    # 根据回测获取最终的调仓日期序列
    BktestParam.refresh_dates = list(filter(lambda x: BktestParam.start_date <= x <= BktestParam.end_date,
                                            BktestParam.refresh_dates))

    BktestParam.monthly_refresh_dates = BktestParam.refresh_dates
    BktestResult.indus_order = cal_indus_order(GlobalParam.yoy_indus, GlobalParam.yoy_base, BktestParam.periods,
                                               BktestParam.window_size, BktestParam.fft_size, BktestParam.gauss_alpha)

    BktestResult.w = cal_long_weight(BktestResult.indus_order, BktestParam.long_weight)
    BktestResult.nav = cal_nav(BktestResult.w, BktestParam, GlobalParam)

    index = np.arange(len(BktestParam.back_dates))
    refresh_index = index[[(i in BktestParam.monthly_refresh_dates) for i in BktestParam.back_dates]]
    base = GlobalParam.base_nav[BktestParam.start_loc: BktestParam.end_loc+1]
    base = base / base[0]
    BktestResult.nav_perf = performance(BktestResult.nav, base, refresh_index)

    # 显示回测结果
    print('------------------------------------')
    print("实际回测开始时间[{}]，结束时间[{}]\n".format(num_date(BktestParam.start_date),
                                           num_date(BktestParam.end_date)))
    print("组合年化{:.2f}%，基准年化{:.2f}%，超额年化{:.2f}%".format(100 * BktestResult.nav_perf.iloc[0, 0],
          100 * BktestResult.nav_perf.iloc[1, 0],
          100 * BktestResult.nav_perf.iloc[0, 4]))

    indus_nav = GlobalParam.daily_close[BktestParam.start_loc: BktestParam.end_loc + 1]
    indus_nav = indus_nav / indus_nav[0, :]

    index = pd.to_datetime(pd.Index([num_date(i) for i in BktestParam.back_dates]))

    mpl.rcParams['font.sans-serif'] = ['FangSong']  # 指定默认字体
    mpl.rcParams['axes.unicode_minus'] = False      # 解决保存图像是负号'-'显示为方块的问题

    plt.plot(index, BktestResult.nav, linewidth=1.0, label='组合', color='red')
    plt.plot(index, base, linewidth=1.0, label='基准', color='blue')
    plt.plot(index, indus_nav, linewidth=0.5)
    labels = ['组合', '基准'] + GlobalParam.indus_name
    plt.legend(labels)
    plt.title("周期因子[{}] 训练长度[{}] 滤波参数[{}]".format(
              BktestParam.periods, BktestParam.window_size, BktestParam.gauss_alpha))
    plt.xlabel("时间")
    plt.ylabel("净值")
    plt.show()


def portfolio():
    """
    运行完策略后，将每一期的预测排名和真实的行业走势写入文件，用于比较
    """
    indus_name = ['上游', '中游', '下游', '金融', '消费', '成长']
    indus_order = BktestResult.indus_order
    predict_dates = BktestParam.predict_dates

    # 获取预测日期对应的收盘价
    index = [(i in predict_dates) for i in GlobalParam.daily_dates]
    monthly_close = GlobalParam.daily_close[index, :]
    monthly_return = monthly_close[1:, ] / monthly_close[:-1, ] - 1

    # 最后一个预测排名无法获得实际的验证结果
    indus_order = indus_order[:-1, ]
    predict_dates = predict_dates[:-1]

    # 将预测结果和实际收益率写入文件
    m, n = indus_order.shape
    columns = ["预测第{}".format(i + 1) for i in range(n)]
    index = [num_date(i) for i in predict_dates]
    full_info = pd.DataFrame(np.empty((m, n)), dtype='str', index=index, columns=columns)
    stat_info = pd.DataFrame(np.empty((m, n)), dtype='int32', index=index, columns=columns)
    for i in range(m):
        # 预测排名
        predict_order = indus_order[i, ]
        # 实际收益率及其排名
        indus_return = monthly_return[i, ]
        index_order = np.argsort(indus_return)[::-1]
        real_order = np.argsort(index_order) + 1
        # 写入文件
        for j in range(n):
            index = list(predict_order).index(j)
            full_info.iloc[i, j] = (indus_name[index] + "({}, {:.2f}%)".format(real_order[index], indus_return[index]))
            stat_info.iloc[i, j] = real_order[index]

    large_name = GlobalParam.indus_name
    columns = ['持仓', '策略', '排名'] + large_name
    index = [num_date(i) for i in predict_dates]
    result = pd.DataFrame(np.empty((m, n+3)), dtype='float64', index=index, columns=columns)
    result.iloc[:, 0].astype('str')
    result.iloc[:, 2].astype('int32')

    for i in range(m):
        # 预测排名
        predict_order = indus_order[i, ]
        # 实际收益率及其排名
        indus_return = monthly_return[i, ]
        index_order = np.argsort(indus_return)[::-1]
        real_order = np.argsort(index_order) + 1
        index = list(predict_order).index(0)
        result.iloc[i, 0] = large_name[index]
        result.iloc[i, 1] = indus_return[index]
        result.iloc[i, 2] = real_order[index]
        result.iloc[i, 3:] = indus_return

    try:
        writer = pd.ExcelWriter("result/持仓明细.xlsx")
        full_info.to_excel(writer, "明细")
        stat_info.to_excel(writer, "排名")
        result.to_excel(writer, "报告")
        writer.save()
    except FileNotFoundError:
        print("在result文件中创建 '持仓明细.xlsx' 后继续此操作")


if __name__ == '__main__':
    sw_indus = pd.read_csv('data/sw_indus.csv', encoding='GBK')
    close = pd.read_csv('data/close.csv', header=None)
    daily_dates = pd.read_csv('data/daily_dates.csv', header=None)
    monthly_dates = pd.read_csv('data/monthly_dates.csv', header=None)
    indus_num = 28
    data = {
        'close': close,
        'indus_num': indus_num,
        'daily_dates': daily_dates,
        'monthly_dates': monthly_dates
    }

    gen_global_param(data, sw_indus)

    bktest_unit()

    # print(GlobalParam.daily_close)
    # print(GlobalParam.daily_dates)
    # print(GlobalParam.base_nav)
    # print(GlobalParam.monthly_indus)
    # print(GlobalParam.monthly_base)
    # print(GlobalParam.monthly_dates)
    # print(GlobalParam.yoy_indus)
    # print(GlobalParam.yoy_base)
    # print(GlobalParam.yoy_dates)

    # print(BktestParam.predict_dates)
    # print(BktestParam.start_date)
    # print(BktestParam.end_date)
    # print(BktestParam.refresh_dates)
    # print(BktestParam.start_loc)
    # print(BktestParam.end_loc)
    # print(BktestParam.back_dates)
    # print(BktestParam.monthly_refresh_dates)
    #
    # print(BktestResult.indus_order)
    # print(BktestResult.w.T)
    # print(BktestResult.nav)
    # print(BktestResult.nav_perf)
