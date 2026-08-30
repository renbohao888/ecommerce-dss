# -*- coding: utf-8 -*-
"""
电子商务大数据分析与智能决策支持系统
基于 Flask 的 Web 系统，覆盖：数据总览、用户行为分析与推荐、商品销售预测、
营销策略优化、供应链风控、商品价格优化 六大模块。
"""
import os
import sys
import time
import threading
from flask import Flask, render_template, request, jsonify

BASE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE)
from ml import models as M  # noqa: E402
from ml import live as LIVE  # noqa: E402

# 实时化配置
LIVE_TICK = 3.0          # 每个实时数据批间隔（秒）
LIVE_BATCH = (2, 5)      # 每批新订单数区间
REFRESH_SEC = 15          # 模型定时重算间隔（秒）

app = Flask(__name__)
app.config['JSON_AS_ASCII'] = False


@app.route('/')
def index():
    data = M.dashboard_data()
    return render_template('dashboard.html', data=data, page='dashboard')


@app.route('/recommend')
def recommend():
    users = M.users()
    all_ids = users['user_id'].tolist()
    default_id = 100
    # 默认挑一个有交互记录的用户
    if default_id not in all_ids:
        default_id = int(all_ids[0])
    user_id = request.args.get('user_id', default=default_id, type=int)
    if user_id not in all_ids:
        user_id = default_id
    recs = M.recommend(user_id, topn=8)
    hist = M.user_history(user_id)
    overview = M.behavior_overview()
    return render_template('recommend.html', recs=recs, hist=hist, overview=overview,
                           user_list=all_ids[:200], user_id=user_id, page='recommend')


@app.route('/forecast')
def forecast():
    days = request.args.get('days', default=14, type=int)
    if days not in (7, 14, 30):
        days = 14
    data = M.forecast(days)
    cat_data = [{'name': c['category'], 'type': 'line', 'smooth': True,
                 'data': c['qty']} for c in data['cat_forecast']]
    return render_template('forecast.html', data=data, cat_data=cat_data,
                           days=days, page='forecast')


@app.route('/marketing')
def marketing():
    data = M.marketing_data()
    return render_template('marketing.html', data=data, page='marketing')


@app.route('/supply')
def supply():
    data = M.supplychain_data()
    return render_template('supply.html', data=data, page='supply')


@app.route('/price')
def price():
    price_data = M.price_data()
    cats = sorted(M.products()['category'].unique())
    # 默认选一个订单样本最丰富的品类，需求曲线更完整
    cat_counts = M.orders().groupby('category').size().sort_values(ascending=False)
    default_cat = cat_counts.index[0]
    cat = request.args.get('cat', default=default_cat)
    if cat not in cats:
        cat = default_cat
    curve = M.price_curve(cat)
    return render_template('price.html', data=price_data, curve=curve, cats=cats, cat=cat, page='price')


@app.route('/service')
def service():
    data = M.customer_service()
    return render_template('service.html', data=data, page='service')



@app.route('/data')
def data_insight():
    data = M.external_data()
    return render_template('data.html', data=data, page='data')


@app.route('/health')
def health():
    from flask import jsonify
    return jsonify(status='ok')


@app.route('/api/live')
def api_live():
    """实时数据流快照：供前端 LIVE 状态栏轮询。"""
    return jsonify(LIVE.snapshot())


@app.route('/api/daily')
def api_daily():
    """最新「近 30 天每日 GMV」序列：供数据总览图自动刷新。"""
    return jsonify(M.dashboard_data()['daily'])


@app.route('/api/chat', methods=['POST'])
def api_chat():
    """AI 数据问答接口（供可拖动悬浮窗调用）。"""
    req = request.get_json(silent=True) or {}
    msg = req.get('message') or ''
    import ml.ai as AI
    try:
        ans = AI.ai_answer(msg)
    except Exception as e:
        ans = {'reply': '抱歉，处理问题出现异常：%s' % e, 'chart': None, 'suggestions': []}
    return jsonify(ans)


def _start_live():
    """启动实时数据流 + 定时重算调度（后台守护线程）。"""
    try:
        LIVE.start(M.products(), M.users(), tick=LIVE_TICK, batch=LIVE_BATCH)
        print('[实时] 实时数据流已启动：每 %.1fs 生成一批新数据。' % LIVE_TICK, flush=True)
    except Exception as e:
        print('[实时] 实时数据流启动失败：%s' % e)

    def _scheduler():
        time.sleep(REFRESH_SEC)
        while True:
            try:
                M.refresh()
                # 打印当次实时快照，便于观察数据在动
                s = LIVE.snapshot()
                if s['orders_total']:
                    print('[实时] 模型已重算 | 实时订单 %d | GMV %.0f | 近1分钟 %d 单'
                          % (s['orders_total'], s['gmv_total'], s['orders_last_min']), flush=True)
            except Exception:
                pass
            time.sleep(REFRESH_SEC)

    t = threading.Thread(target=_scheduler, daemon=True)
    t.start()


if __name__ == '__main__':
    print('启动电子商务大数据分析与智能决策支持系统 ...', flush=True)
    _start_live()
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', '5000')), debug=False)
