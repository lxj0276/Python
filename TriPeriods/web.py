from flask import Flask, request, make_response
import TriPeriods as Tri
import pandas as pd
app = Flask(__name__)


def init_tri():
    sw_indus = pd.read_csv('data/sw_indus.csv', encoding='GBK')
    close = pd.read_csv('mydata/close.csv', header=None)
    daily_dates = pd.read_csv('mydata/daily_dates.csv', header=None)
    monthly_dates = pd.read_csv('mydata/monthly_dates.csv', header=None)
    indus_num = len(sw_indus)
    data = {
        'close': close,
        'indus_num': indus_num,
        'daily_dates': daily_dates,
        'monthly_dates': monthly_dates
    }

    Tri.gen_global_param(data, sw_indus)
    Tri.bktest_unit()


@app.route('/result')
def get_data():

    res = make_response(Tri.BktestResult.df.T.to_json(orient='values'))
    # res = make_response(df.to_json(orient='values'))

    res.headers['Access-Control-Allow-Origin'] = '*'
    res.headers['Access-Control-Allow-Methods'] = 'POST, GET, OPTIONS'
    res.headers['Access-Control-Allow-Headers'] = 'x-requested-with,content-type'

    return res


@app.route('/performance')
def get_performance():
    Tri.BktestResult.nav_perf.columns = ['return', 'volatility', 'sharp', 'drawdown',
                                         'excess', 'excess_vol', 'ic', 'win', 'excess_draw']

    Tri.BktestResult.nav_perf.index = ['p', 'b']
    Tri.BktestResult.nav_perf = Tri.BktestResult.nav_perf.applymap(lambda x: round(x, 3))

    res = make_response(Tri.BktestResult.nav_perf.to_json())
    # res = make_response(df.to_json(orient='values'))

    res.headers['Access-Control-Allow-Origin'] = '*'
    res.headers['Access-Control-Allow-Methods'] = 'POST, GET, OPTIONS'
    res.headers['Access-Control-Allow-Headers'] = 'x-requested-with,content-type'

    return res


@app.route('/back_test', methods=['POST'])
def back_test():
    if request.method == 'POST' and request.form:

        content = request.form.to_dict()

        res = make_response("hello")

        res.headers['Access-Control-Allow-Origin'] = '*'
        res.headers['Access-Control-Allow-Methods'] = 'POST'
        res.headers['Access-Control-Allow-Headers'] = 'x-requested-with,content-type'

        Tri.BktestParam.init_period = [content['begin'], content['end']]
        Tri.BktestParam.commission_rate = float(content['commission'])
        Tri.BktestParam.periods = [int(i) for i in content['period'].split(',')]
        Tri.BktestParam.window_size = int(content['window'])
        Tri.BktestParam.fft_size = int(content['fft'])
        Tri.BktestParam.gauss_alpha = int(content['gauss'])
        Tri.BktestParam.long_weight = [float(i) for i in content['long'].split(',')]

        Tri.bktest_unit()

        return res


if __name__ == '__main__':
    init_tri()
    app.run()
