# -*- coding: utf-8 -*-
"""
系统内实时数据流引擎（方案A）：
后台线程按固定节奏（默认每 3 秒）生成一批新的订单/浏览行为/评价，追加到内存中的
实时数据表中，使 dashboard、推荐、预测、营销、价格、风控等分析结果"随时间自动更新"，
并对外提供实时 KPI 快照（/api/live）。

说明：
- 数据在系统进程内实时刷新（模拟"活跃"电商流量），无需外部爬取，稳定且不违反平台规则。
- 不影响基础 CSV 数据；实时行仅在内存中累积，并设缓冲上限防止无限增长。
- 未调用 start() 时，所有访问函数返回空表，系统行为与静态版完全一致（便于测试）。
"""
import threading
import time

import numpy as np
import pandas as pd

# ---- 可配置参数 ----
DEFAULT_TICK = 3.0            # 每个数据批的间隔（秒）
DEFAULT_BATCH = (2, 5)        # 每批新订单数（随机整数区间）
MAX_ORDERS = 800              # 内存中最多保留的实时订单数
MAX_BEHAVIORS = 600
MAX_REVIEWS = 300
HIST_SIZE = 120               # 实时历史滚动点数（供迷你图/趋势）

ORD_COLS = ['order_id', 'user_id', 'product_id', 'category', 'qty', 'unit_price',
            'paid_amount', 'discount', 'order_date', 'channel', 'warehouse_dist',
            'city_level', 'delivery_days', 'status']
BEH_COLS = ['user_id', 'product_id', 'behavior_type', 'timestamp']
REV_COLS = ['order_id', 'user_id', 'product_id', 'category', 'rating', 'sentiment',
            'review_date']

_state = {
    'running': False,
    'started_at': None,
    'last_tick': None,
    'seq': 0,
    'orders': None, 'behaviors': None, 'reviews': None,
    'hist': [],
}
_lock = threading.Lock()
_thread = None
_tick = DEFAULT_TICK
_bmin, _bmax = DEFAULT_BATCH
_prod = None
_users = None
_rng = np.random.default_rng()


def _now():
    return pd.Timestamp.now()


def _gen_orders(k):
    """生成 k 条实时订单，字段结构与基础 orders.csv 一致。"""
    rows = []
    users = _users
    prods = _prod
    for _ in range(k):
        usr = users.sample(1).iloc[0]
        cat = str(usr['fav_category'])
        cand = prods[prods['category'] == cat]
        if cand.empty:
            cand = prods
        p = cand.sample(1).iloc[0]
        base_price = float(p['price'])
        disc = float(_rng.choice([1.0, 0.9, 0.8, 0.7, 0.6], p=[0.3, 0.3, 0.2, 0.15, 0.05]))
        unit_price = round(base_price * disc, 2)
        qty = max(1, int(round(3.2 - unit_price / 1200.0 + _rng.normal(0, 0.4))))
        channel = str(_rng.choice(['App', 'Web', 'MiniProgram'], p=[0.6, 0.25, 0.15]))
        wd = float(usr['warehouse_dist'])
        cl = int(usr['city_level'])
        arrival = max(0, int(np.round(_rng.normal(2 + wd / 12.0 + cl * 0.3, 0.8))))
        status = 'delivered' if arrival <= 5 else 'delayed'
        _state['seq'] += 1
        oid = 'LVO%08d' % _state['seq']
        rows.append({
            'order_id': oid, 'user_id': int(usr['user_id']), 'product_id': p['product_id'],
            'category': p['category'], 'qty': qty, 'unit_price': unit_price,
            'paid_amount': round(unit_price * qty, 2),
            'discount': round(1 - unit_price / base_price, 3),
            'order_date': _now(), 'channel': channel,
            'warehouse_dist': wd, 'city_level': cl,
            'delivery_days': arrival, 'status': status,
        })
    return pd.DataFrame(rows, columns=ORD_COLS)

def _gen_behaviors(k):
    """生成 k*2 条实时浏览/加购行为，供推荐交互矩阵持续变化。"""
    rows = []
    users = _users
    prods = _prod
    for _ in range(k * 2):
        usr = users.sample(1).iloc[0]
        p = prods.sample(1).iloc[0]
        bt = str(_rng.choice(['view', 'cart', 'purchase'], p=[0.7, 0.2, 0.1]))
        rows.append({'user_id': int(usr['user_id']), 'product_id': p['product_id'],
                     'behavior_type': bt, 'timestamp': _now()})
    return pd.DataFrame(rows, columns=BEH_COLS)


def _gen_reviews(orders_df):
    """针对部分新订单生成实时评价（驱动满意度/负面评价归因变化）。"""
    rows = []
    if orders_df is None or not len(orders_df):
        return pd.DataFrame(columns=REV_COLS)
    for _, r in orders_df.iterrows():
        if _rng.random() < 0.5:
            continue
        base = 4.0
        if r['delivery_days'] > 5:
            base -= 1.5
        base += float(r['city_level']) / 5.0 - 0.2
        rating = int(np.clip(round(base + _rng.normal(0, 0.6)), 1, 5))
        sentiment = 'positive' if rating >= 4 else ('neutral' if rating == 3 else 'negative')
        rows.append({'order_id': r['order_id'], 'user_id': r['user_id'],
                     'product_id': r['product_id'], 'category': r['category'],
                     'rating': rating, 'sentiment': sentiment, 'review_date': _now()})
    return pd.DataFrame(rows, columns=REV_COLS)


def _loop():
    """后台线程主循环：每间隔生成一批数据并维护实时快照。"""
    while _state['running']:
        time.sleep(_tick)
        try:
            k = int(_rng.integers(_bmin, _bmax + 1))
            ob = _gen_orders(k)
            bb = _gen_behaviors(k)
            rr = _gen_reviews(ob)
            with _lock:
                if _state['orders'] is None:
                    _state['orders'] = ob
                else:
                    _state['orders'] = pd.concat([_state['orders'], ob], ignore_index=True)
                if _state['behaviors'] is None:
                    _state['behaviors'] = bb
                else:
                    _state['behaviors'] = pd.concat([_state['behaviors'], bb], ignore_index=True)
                if len(rr):
                    if _state['reviews'] is None:
                        _state['reviews'] = rr
                    else:
                        _state['reviews'] = pd.concat([_state['reviews'], rr], ignore_index=True)
                # 滚动缓冲上限
                if len(_state['orders']) > MAX_ORDERS:
                    _state['orders'] = _state['orders'].tail(MAX_ORDERS)
                if len(_state['behaviors']) > MAX_BEHAVIORS:
                    _state['behaviors'] = _state['behaviors'].tail(MAX_BEHAVIORS)
                if len(_state['reviews']) > MAX_REVIEWS:
                    _state['reviews'] = _state['reviews'].tail(MAX_REVIEWS)
                # 本批聚合，追加到历史
                _state['hist'].append({
                    't': _now().isoformat(),
                    'o': int(len(ob)),
                    'g': round(float(ob['paid_amount'].sum()), 2),
                    'd': int((ob['status'] == 'delayed').sum()),
                })
                _state['hist'] = _state['hist'][-HIST_SIZE:]
                _state['last_tick'] = _now().time().strftime('%H:%M:%S')
        except Exception:
            time.sleep(1)

def start(products, users, tick=DEFAULT_TICK, batch=DEFAULT_BATCH):
    """启动实时数据流线程（幂等）。products/users 用于生成真实规格的实时行。"""
    global _thread, _tick, _bmin, _bmax, _prod, _users
    if _state['running']:
        return True
    _prod = products
    _users = users
    _tick = float(tick)
    if isinstance(batch, (tuple, list)) and len(batch) == 2:
        _bmin, _bmax = int(batch[0]), int(batch[1])
    _state['orders'] = None
    _state['behaviors'] = None
    _state['reviews'] = None
    _state['hist'] = []
    _state['seq'] = 0
    _state['started_at'] = _now()
    _state['running'] = True
    _thread = threading.Thread(target=_loop, daemon=True)
    _thread.start()
    return True


def stop():
    """停止实时数据流线程。"""
    _state['running'] = False


def is_running():
    return bool(_state['running'])


def orders():
    """实时订单（空表或有效 DataFrame；已 start 后可用）。"""
    with _lock:
        df = _state['orders']
        return df.copy() if df is not None and len(df) else pd.DataFrame(columns=ORD_COLS)


def behaviors():
    with _lock:
        df = _state['behaviors']
        return df.copy() if df is not None and len(df) else pd.DataFrame(columns=BEH_COLS)


def reviews():
    with _lock:
        df = _state['reviews']
        return df.copy() if df is not None and len(df) else pd.DataFrame(columns=REV_COLS)


def snapshot():
    """实时 KPI 快照，供 /api/live 与前端状态栏轮询。"""
    with _lock:
        ods = _state['orders']
        n = len(ods) if ods is not None else 0
        gmv = float(ods['paid_amount'].sum()) if n else 0.0
        now = _now()
        last_min = ods[ods['order_date'] >= now - pd.Timedelta(minutes=1)] if n else None
        n_min = len(last_min) if n else 0
        gmv_min = float(last_min['paid_amount'].sum()) if n else 0.0
        delay_min = int((last_min['status'] == 'delayed').sum()) if n else 0
        hist = [{'t': h['t'], 'o': h['o'], 'g': h['g'], 'd': h['d']} for h in _state['hist']]
        return {
            'running': bool(_state['running']),
            'tick': _tick,
            'last_tick': _state['last_tick'],
            'orders_total': n,
            'gmv_total': round(gmv, 2),
            'avg_order': round(gmv / n, 2) if n else 0.0,
            'orders_last_min': n_min,
            'gmv_last_min': round(gmv_min, 2),
            'delayed_last_min': delay_min,
            'delay_rate_last_min': round(delay_min / n_min * 100, 2) if n_min else None,
            'hist': hist,
        }

