# -*- coding: utf-8 -*-
"""机器学习/数据分析模型层：负责加载数据、训练模型、产出页面所需的图表数据。"""
import os
import threading
import urllib.request
import numpy as np
import pandas as pd

from ml import live as LIVE

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(BASE, 'data')
_cache = {}

# 模型训练/重算互斥锁：避免“后台定时重算”与“页面请求触发的按需训练”并发导致的双重训练/竞态崩溃，
# 同时保证请求线程读到的始终是一个完整可用的模型对象。
_train_lock = threading.RLock()


def load(name):
    if name not in _cache:
        _cache[name] = pd.read_csv(os.path.join(DATA, name + '.csv'))
    return _cache[name]


def users():
    return load('users')


def products():
    return load('products')


def orders():
    o = load('orders')
    o['order_date'] = pd.to_datetime(o['order_date'])
    live = LIVE.orders()
    if live is not None and len(live):
        o = pd.concat([o, live], ignore_index=True)
    return o


def behaviors():
    b = load('behaviors')
    live = LIVE.behaviors()
    if live is not None and len(live):
        b = pd.concat([b, live], ignore_index=True)
    return b


def reviews():
    r = load('reviews')
    live = LIVE.reviews()
    if live is not None and len(live):
        r = pd.concat([r, live], ignore_index=True)
    return r


def refresh():
    """定时/启动时在后台重算所有模型缓存，使结果随最新实时数据自动更新。

    与旧版的区别：不再把缓存置空（那会迫使下一个页面请求在“请求线程”里同步重训，
    造成页面跳转缓慢，并存在并发置空竞态）。这里改为“保温重算”——每个模型在锁内
    重建并整体替换，请求线程读到的始终是完整可用的模型，因此页面导航始终保持快速。
    """
    for trainer in (train_recommend, train_forecast, train_marketing,
                    train_supplychain, train_price):
        try:
            with _train_lock:
                trainer()
        except Exception as e:
            print('[模型] %s 重算失败：%s' % (getattr(trainer, '__name__', trainer), e), flush=True)


def _num(x, nd=2):
    """numpy -> python float，便于 JSON 输出。"""
    if x is None:
        return None
    try:
        if pd.isna(x):
            return None
    except Exception:
        pass
    return round(float(x), nd)


def dashboard_data():
    """数据总览：KPI + 各维度图表数据。"""
    u = users()
    p = products()
    o = orders()
    r = reviews()
    gmv = o['paid_amount'].sum()

    # 日销售额序列（最近 30 天）
    d = o.groupby(o['order_date'].dt.date)['paid_amount'].agg(['sum', 'count']).reset_index()
    d.columns = ['date', 'gmv', 'orders']
    d = d.sort_values('date').tail(30)

    cat_sales = o.groupby('category')['paid_amount'].sum().sort_values(ascending=False)
    channels = o.groupby('channel')['paid_amount'].sum().sort_values(ascending=False)
    top_prods = o.groupby(['product_id', 'category'])['paid_amount'].sum().sort_values(ascending=False).head(8)
    status = o['status'].value_counts()
    sat = r['rating'].value_counts().sort_index()

    delay_rate = float((o['status'] == 'delayed').mean())

    # 结合评价与物流：各评分档位的配送延迟率
    ro = r.merge(o[['order_id', 'status']], on='order_id', how='left')
    rating_delay = ro.groupby('rating')['status'].apply(lambda s: float((s == 'delayed').mean()) * 100)

    kpis = {
        'users': int(len(u)), 'products': int(len(p)), 'orders': int(len(o)),
        'gmv': _num(gmv), 'avg_order': _num(gmv / len(o)),
        'avg_sat': _num(r['rating'].mean()), 'complaints': int(u['complaints'].sum()),
        'delay_rate': _num(delay_rate * 100), 'avg_delivery': _num(o['delivery_days'].mean()),
        'reviews': int(len(r)),
    }

    return {
        'kpis': kpis,
        'daily': {'date': [str(x) for x in d['date']],
                  'gmv': [_num(x) for x in d['gmv']],
                  'orders': [_num(x, 0) for x in d['orders']]},
        'cat_sales': [{'name': k, 'value': _num(v)} for k, v in cat_sales.items()],
        'channels': [{'name': k, 'value': _num(v)} for k, v in channels.items()],
        'top_prods': [{'name': k[0], 'category': k[1], 'value': _num(v)} for k, v in top_prods.items()],
        'status': [{'name': k, 'value': int(v)} for k, v in status.items()],
        'satisfaction': [{'rating': int(k), 'count': int(v)} for k, v in sat.items()],
        'rating_delay': [{'rating': int(k), 'delay': _num(v, 2)} for k, v in rating_delay.items()],
    }


# ---------------- 用户行为分析与推荐（基于物品的协同过滤）----------------
_user_item = None  # 缓存交互矩阵与相似度


def _build_interaction():
    b = behaviors().copy()
    b['w'] = b['behavior_type'].map({'view': 1, 'cart': 2, 'purchase': 3})
    o = orders()[['user_id', 'product_id']].copy()
    o['w'] = 3
    both = pd.concat([b[['user_id', 'product_id', 'w']], o[['user_id', 'product_id', 'w']]])
    agg = both.groupby(['user_id', 'product_id'])['w'].sum().reset_index()
    piv = agg.pivot(index='user_id', columns='product_id', values='w').fillna(0)
    return piv


def train_recommend():
    global _user_item
    from sklearn.metrics.pairwise import cosine_similarity
    piv = _build_interaction()
    X = piv.values.T  # 物品 x 用户
    sim = cosine_similarity(X)  # 物品相似度矩阵
    top = orders().groupby('product_id')['qty'].sum().sort_values(ascending=False)
    _user_item = {'piv': piv, 'sim': sim, 'items': list(piv.columns),
                  'users': list(piv.index), 'top': top}
    return _user_item


def recommend(user_id, topn=8):
    """基于物品相似度为 user_id 推荐 topn 个商品。"""
    global _user_item
    if _user_item is None:
        with _train_lock:
            if _user_item is None:
                train_recommend()
    piv, sim, items = _user_item['piv'], _user_item['sim'], _user_item['items']
    prod = products().set_index('product_id')

    if user_id not in piv.index:
        ids = _user_item['top'].head(topn).index.tolist()
        recs = []
        for pid in ids:
            r = prod.loc[pid]
            recs.append({'id': pid, 'name': r['name'], 'category': r['category'],
                         'price': _num(r['price']), 'reason': '平台热销', 'score': 0})
        return recs

    user_vec = piv.loc[user_id].values.astype(float)
    scores = sim.dot(user_vec)
    interacted = user_vec > 0
    cand = np.where(interacted, -1.0, scores)  # 屏蔽已交互商品
    order = np.argsort(cand)[::-1]
    recs = []
    liked = np.where(user_vec > 0)[0]
    for idx in order:
        if cand[idx] <= 0:
            break
        pid = items[idx]
        r = prod.loc[pid]
        base = liked[np.argmax([sim[idx, j] for j in liked])] if len(liked) else -1
        reason = f'与您喜欢的「{prod.loc[items[base]]["name"]}」相似'
        recs.append({'id': pid, 'name': r['name'], 'category': r['category'],
                     'price': _num(r['price']), 'reason': reason, 'score': _num(scores[idx], 3)})
        if len(recs) >= topn:
            break
    if not recs:
        ids = _user_item['top'].head(topn).index.tolist()
        for pid in ids:
            r = prod.loc[pid]
            recs.append({'id': pid, 'name': r['name'], 'category': r['category'],
                         'price': _num(r['price']), 'reason': '平台热销', 'score': 0})
    return recs


def user_history(user_id):
    """某用户的浏览/加购/购买记录、画像。"""
    b = behaviors()
    bh = b[b['user_id'] == user_id].groupby(['product_id', 'behavior_type'])['behavior_type'].count().reset_index(name='n')
    u = users().set_index('user_id')
    prof = {}
    if user_id in u.index:
        row = u.loc[user_id]
        prof = {'device': row['device'], 'city_level': int(row['city_level']),
                'gender': row['gender'], 'age_group': int(row['age_group']),
                'hours_app': int(row['hours_app']), 'orders_last_month': int(row['orders_last_month']),
                'days_since_order': int(row['days_since_order']), 'fav_category': row['fav_category'],
                'satisfaction': int(row['satisfaction']), 'complaints': int(row['complaints']),
                'coupons_used': int(row['coupons_used'])}
    prod = products().set_index('product_id')
    hist = []
    for pid, g in bh.groupby('product_id'):
        if pid not in prod.index:
            continue
        r = prod.loc[pid]
        gv = g.set_index('behavior_type')['n']
        hist.append({'id': pid, 'name': r['name'], 'category': r['category'],
                     'price': _num(r['price']), 'view': int(gv.get('view', 0)),
                     'cart': int(gv.get('cart', 0)), 'buy': int(gv.get('purchase', 0))})
    return {'profile': prof, 'history': hist}


# ---------------- 商品销售预测（随机森林时序回归）----------------
_forecast = None


def _time_x(dates, trend_start):
    d = pd.Series(pd.to_datetime(dates)).reset_index(drop=True)
    return np.column_stack([d.dt.weekday, (d.dt.weekday >= 5).astype(int),
                            d.dt.month, d.dt.day, np.arange(trend_start, trend_start + len(d))]).astype(float)


def train_forecast():
    global _forecast
    from sklearn.ensemble import RandomForestRegressor
    o = orders().copy()
    o['d'] = o['order_date'].dt.date
    daily = o.groupby('d').agg(gmv=('paid_amount', 'sum'), qty=('qty', 'sum'),
                               orders=('order_id', 'count')).reset_index().rename(columns={'d': 'date'})
    daily['date'] = pd.to_datetime(daily['date'])
    daily = daily.sort_values('date').reset_index(drop=True)

    X = _time_x(daily['date'], 0)
    model = RandomForestRegressor(n_estimators=150, random_state=0, n_jobs=-1)
    model.fit(X, daily['gmv'].values)
    pred = model.predict(X)
    mae = float(np.mean(np.abs(pred - daily['gmv'].values)))
    rmse = float(np.sqrt(np.mean((pred - daily['gmv'].values) ** 2)))

    cats = {}
    for c in sorted(o['category'].unique()):
        sub = o[o['category'] == c].groupby('d')['qty'].sum().reset_index(name='qty').rename(columns={'d': 'date'})
        sub['date'] = pd.to_datetime(sub['date'])
        sub = sub.sort_values('date').reset_index(drop=True)
        Xc = _time_x(sub['date'], 0)
        m = RandomForestRegressor(n_estimators=100, random_state=0, n_jobs=-1)
        m.fit(Xc, sub['qty'].values)
        cats[c] = {'model': m, 'nrows': len(sub), 'series': sub}

    _forecast = {'daily': daily, 'model': model, 'nrows': len(daily),
                 'cats': cats, 'mae': mae, 'rmse': rmse, 'last_date': daily['date'].max()}
    return _forecast


def forecast(days=14):
    global _forecast
    if _forecast is None:
        with _train_lock:
            if _forecast is None:
                train_forecast()
    daily, model, nrows, mae = _forecast['daily'], _forecast['model'], _forecast['nrows'], _forecast['mae']
    last = daily['date'].max()
    fut = pd.date_range(last + pd.Timedelta(days=1), periods=days)
    yhat = np.maximum(model.predict(_time_x(fut, nrows)), 0)

    cat_forecast = []
    for c, info in _forecast['cats'].items():
        fx = np.maximum(info['model'].predict(_time_x(fut, info['nrows'])), 0)
        cat_forecast.append({'category': c, 'qty': [_num(x, 0) for x in fx]})

    return {'dates': [str(x.date()) for x in fut], 'forecast': [_num(x, 0) for x in yhat],
            'cat_forecast': cat_forecast, 'mae': _num(mae), 'rmse': _num(_forecast['rmse']),
            'last_date': str(last.date()),
            'history': {'date': [str(x.date()) for x in daily['date']],
                        'gmv': [_num(x, 0) for x in daily['gmv']]}}


# ---------------- 营销策略优化（RFM 客户分层 + 聚类）----------------
_marketing = None


def train_marketing():
    global _marketing
    from sklearn.preprocessing import StandardScaler
    from sklearn.cluster import KMeans
    o = orders()
    ref = o['order_date'].max()
    rfm = o.groupby('user_id').agg(R=('order_date', 'max'), F=('order_id', 'count'),
                                   M=('paid_amount', 'sum')).reset_index()
    rfm['R'] = (ref - rfm['R']).dt.days
    X = StandardScaler().fit_transform(rfm[['R', 'F', 'M']])
    km = KMeans(n_clusters=4, random_state=0, n_init=10)
    km.fit(X)
    rfm['segment'] = km.labels_

    info = rfm.groupby('segment').agg(R=('R', 'mean'), F=('F', 'mean'), M=('M', 'mean'))
    high_r = info['R'].idxmax()
    rest = [c for c in info.index if c != high_r]
    high_m = info.loc[rest, 'M'].idxmax()
    rest2 = [c for c in rest if c != high_m]
    high_f = info.loc[rest2, 'F'].idxmax()
    low = [c for c in rest2 if c != high_f][0]
    names = {high_r: '流失风险客户', high_m: '高价值客户', high_f: '活跃客户', low: '潜力客户'}
    rfm['seg_name'] = rfm['segment'].map(names)

    u = users().set_index('user_id')
    rfm['satisfaction'] = rfm['user_id'].map(u['satisfaction'])
    rfm['complaints'] = rfm['user_id'].map(u['complaints'])
    rfm['coupons'] = rfm['user_id'].map(u['coupons_used'])

    seg_strategy = {
        '高价值客户': '定向高额满减券 + VIP 专属客服 + 新品优先购',
        '活跃客户': '限时折扣券 + 会员积分翻倍 + 主播专场提醒',
        '潜力客户': '新人/品类首单立减券 + 精准品类推荐',
        '流失风险客户': '挽回大额券 + 短信触达 + 老客回访',
    }
    segs = []
    for name, g in rfm.groupby('seg_name'):
        segs.append({'name': name, 'count': int(len(g)), 'pct': _num(len(g) / len(rfm) * 100),
                     'avgR': _num(g['R'].mean()), 'avgF': _num(g['F'].mean(), 1),
                     'avgM': _num(g['M'].mean()), 'sat': _num(g['satisfaction'].mean()),
                     'comp': _num(g['complaints'].mean()), 'coupon': _num(g['coupons'].mean()),
                     'strategy': seg_strategy[name]})

    scat = rfm.sample(min(600, len(rfm)), random_state=1)
    scatter = [{'r': _num(r), 'm': _num(m), 'seg': seg} for r, m, seg in
               zip(scat['R'], scat['M'], scat['seg_name'])]
    # 客户服务洞察
    service = {
        'avg_sat': _num(rfm['satisfaction'].mean()), 'avg_comp': _num(rfm['complaints'].mean()),
        'avg_coupon': _num(rfm['coupons'].mean()),
        'sat_rate': _num((u['satisfaction'] >= 4).mean() * 100),
    }
    _marketing = {'rfm': rfm, 'segs': segs, 'scatter': scatter, 'service': service,
                  'names': names, 'seg_strategy': seg_strategy}
    return _marketing


def marketing_data():
    global _marketing
    if _marketing is None:
        with _train_lock:
            if _marketing is None:
                train_marketing()
    d = {
        'segs': _marketing['segs'], 'scatter': _marketing['scatter'],
        'service': _marketing['service'],
    }
    # 订单量前 5 的用户（用于示例表格）
    rfm = _marketing['rfm'].sort_values('M', ascending=False).head(8)
    d['top_users'] = [{'user_id': int(r['user_id']), 'R': _num(r['R']), 'F': _num(r['F'], 1),
                       'M': _num(r['M']), 'seg': r['seg_name']} for _, r in rfm.iterrows()]
    return d


# ---------------- 供应链韧性监控（随机森林风险分类）----------------
_supply = None


def train_supplychain():
    global _supply
    from sklearn.ensemble import RandomForestClassifier
    o = orders().copy()
    o['is_delayed'] = (o['status'] == 'delayed').astype(int)
    o['month'] = o['order_date'].dt.month
    o['cat_code'] = o['category'].astype('category').cat.codes
    o['channel_code'] = o['channel'].astype('category').cat.codes
    feats = ['warehouse_dist', 'city_level', 'qty', 'unit_price', 'discount',
             'month', 'cat_code', 'channel_code']
    X = o[feats].values
    y = o['is_delayed'].values
    clf = RandomForestClassifier(n_estimators=200, random_state=0, n_jobs=-1)
    clf.fit(X, y)
    o['risk'] = clf.predict_proba(X)[:, 1]
    imp = sorted(zip(feats, clf.feature_importances_), key=lambda t: -t[1])

    o['dist_bin'] = pd.cut(o['warehouse_dist'], [0, 10, 20, 30, 60], labels=['<10', '10-20', '20-30', '30-60'])
    by_dist = o.groupby('dist_bin', observed=True)['is_delayed'].mean() * 100
    by_cat = o.groupby('category')['is_delayed'].mean() * 100
    top_risk = o.nlargest(20, 'risk')

    _supply = {'clf': clf, 'feats': feats, 'imp': imp, 'o': o,
               'by_dist': [{'name': str(k), 'value': _num(v, 1)} for k, v in by_dist.items()],
               'by_cat': [{'name': k, 'value': _num(v, 1)} for k, v in by_cat.sort_values(ascending=False).items()],
               'delay_rate': _num((o['is_delayed'].mean()) * 100),
               'top': [{'order_id': r['order_id'], 'user_id': int(r['user_id']),
                        'product_id': r['product_id'], 'category': r['category'],
                        'dist': _num(r['warehouse_dist']), 'delivery': int(r['delivery_days']),
                        'risk': _num(r['risk'] * 100, 1)} for _, r in top_risk.iterrows()]}
    return _supply


def supplychain_data():
    global _supply
    if _supply is None:
        with _train_lock:
            if _supply is None:
                train_supplychain()
    return {'imp': [{'name': k, 'value': _num(v, 4)} for k, v in _supply['imp']],
            'by_dist': _supply['by_dist'], 'by_cat': _supply['by_cat'],
            'delay_rate': _supply['delay_rate'], 'top': _supply['top']}


# ---------------- 商品价格优化（价格弹性估计与最优定价）----------------
_price = None


def train_price():
    global _price
    o = orders()
    prod = products().set_index('product_id')
    rows = []
    for pid, g in o.groupby('product_id'):
        g = g[g['unit_price'] > 0]
        if len(g) < 8:
            continue
        agg = g.groupby('unit_price').agg(qty=('qty', 'sum')).reset_index()
        agg = agg[agg['qty'] > 0]
        if len(agg) < 3 or pid not in prod.index:
            continue
        x = np.log(agg['unit_price'].values)
        y = np.log(agg['qty'].values)
        if np.std(x) == 0:
            continue
        e = float(np.clip(np.polyfit(x, y, 1)[0], -3.0, -0.2))
        r = prod.loc[pid]
        price = float(r['price'])
        cost = float(r['cost'])
        category = r['category']

        if e <= -1.01:
            pstar = cost * e / (1 + e)          # 垄断最优定价
        elif e < -0.5:
            pstar = price * 0.95                # 缺乏弹性：小幅降价扩量
        else:
            pstar = price * 1.08                # 无弹性：适当提价
        pstar = float(max(cost * 1.05, min(pstar, price * 1.5)))

        avg_qty = float(g['qty'].mean())
        new_qty = avg_qty * (pstar / price) ** e
        cur_rev = price * avg_qty
        new_rev = pstar * new_qty
        rows.append({'id': pid, 'name': r['name'], 'category': category,
                     'price': _num(price), 'cost': _num(cost), 'elasticity': _num(e, 3),
                     'optimal': _num(pstar), 'cur_rev': _num(cur_rev), 'new_rev': _num(new_rev),
                     'delta_rev': _num(new_rev - cur_rev), 'pct': _num((pstar / price - 1) * 100, 1)})
    df = pd.DataFrame(rows)
    if len(df):
        df = df.sort_values('delta_rev', ascending=False)
    _price = {'rows': df.to_dict('records'), 'count': int(len(df))}
    return _price


def price_data():
    global _price
    if _price is None:
        with _train_lock:
            if _price is None:
                train_price()
    rows = _price['rows']
    if rows:
        avg_el = sum(r['elasticity'] for r in rows) / len(rows)
        gains = sum(1 for r in rows if r['delta_rev'] > 0)
        total_gain = sum(r['delta_rev'] for r in rows if r['delta_rev'] > 0)
    else:
        avg_el = 0.0
        gains = 0
        total_gain = 0.0
    return {'rows': rows, 'count': _price['count'], 'avg_elasticity': _num(avg_el),
            'gains': gains, 'total_gain': _num(total_gain)}


def price_curve(category):
    """某类别的价格-需求曲线（分箱拟合）。"""
    o = orders()
    g = o[o['category'] == category]
    a = g.groupby('unit_price').agg(qty=('qty', 'mean')).reset_index()
    a = a[a['qty'] > 0].sort_values('unit_price')
    curve = [{'price': _num(p), 'qty': _num(q, 3)} for p, q in zip(a['unit_price'], a['qty'])]
    e = -1.0
    if len(a) >= 3 and np.std(np.log(a['unit_price'].values)) > 0:
        e = float(np.clip(np.polyfit(np.log(a['unit_price'].values),
                                     np.log(a['qty'].values), 1)[0], -3, -0.2))
    return {'category': category, 'elasticity': _num(e, 3), 'curve': curve}


# ---------------- 用户行为概览 ----------------
def behavior_overview():
    b = behaviors().copy()
    b['timestamp'] = pd.to_datetime(b['timestamp'])
    b['hour'] = b['timestamp'].dt.hour
    counts = b['behavior_type'].value_counts()
    pc = b.merge(products()[['product_id', 'category']], on='product_id')
    top_cat = pc.groupby('category')['behavior_type'].count().sort_values(ascending=False).head(6)
    hourly = b.groupby('hour').size().reindex(range(24), fill_value=0)
    return {
        'total': int(len(b)),
        'counts': {'view': int(counts.get('view', 0)), 'cart': int(counts.get('cart', 0)),
                   'purchase': int(counts.get('purchase', 0))},
        'top_cat': [{'name': k, 'value': int(v)} for k, v in top_cat.items()],
        'hourly': [{'hour': int(h), 'value': int(v)} for h, v in hourly.items()],
    }


# ---------------- 统一训练入口 ----------------
def build_all():
    """训练并缓存所有模型，返回耗时信息。"""
    import time
    t0 = time.time()
    train_recommend()
    train_forecast()
    train_marketing()
    train_supplychain()
    train_price()
    return {'status': 'ok', 'seconds': round(time.time() - t0, 2)}


# ---------------- 客户服务优化 ----------------
def customer_service():
    """客户服务优化：满意度、投诉画像、负面评价归因、风险客户与策略建议。"""
    u = users()
    r = reviews()
    o = orders()

    # 满意度与投诉指标
    avg_sat = float(u['satisfaction'].mean())
    total_comp = int(u['complaints'].sum())
    comp_users = int((u['complaints'] > 0).sum())
    comp_rate = comp_users / len(u) * 100

    sentiment = r['sentiment'].value_counts()
    pos = int(sentiment.get('positive', 0))
    neg = int(sentiment.get('negative', 0))
    neu = int(sentiment.get('neutral', 0))
    neg_rate = neg / len(r) * 100
    avg_rating = float(r['rating'].mean())

    # 各维度投诉画像
    c_dev = u.groupby('device')['complaints'].sum().sort_values(ascending=False)
    c_city = u.groupby('city_level')['complaints'].sum().sort_values(ascending=False)
    c_age = u.groupby('age_group')['complaints'].sum().sort_values(ascending=False)

    # 负面评价归因（按品类）
    neg_rev = r[r['sentiment'] == 'negative']
    ncat = neg_rev.groupby('category').size().sort_values(ascending=False).head(8)
    # 各品类平均评分
    sat_cat = r.groupby('category')['rating'].mean().sort_values(ascending=True).head(8)
    # 各品类延迟率
    delay_cat = o.groupby('category')['status'].apply(lambda s: float((s == 'delayed').mean()) * 100)

    # 负面评价 + 延迟率 组合
    combo = []
    for c in neg_rev['category'].unique():
        combo.append({'category': c, 'neg': int((neg_rev['category'] == c).sum()),
                      'delay': _num(delay_cat.get(c, 0), 2)})
    combo = sorted(combo, key=lambda x: x['neg'], reverse=True)[:8]

    # 评分x投诉风险打分
    u2 = u.copy()
    u2['risk'] = u2['complaints'] * 2 + (5 - u2['satisfaction']) + u2['orders_last_month'] * 0.1
    top = u2.sort_values('risk', ascending=False).head(10)

    # 服务策略建议
    suggestions = []
    if comp_rate > 8:
        suggestions.append('投诉率偏高，建议上线智能客服工单（自动分类/语义检索），提升响应速度。')
    if neg_rate > 12:
        suggestions.append('负面评价占比偏高，建议对差评品类采取主动外呼回访与售后关怀。')
    longest_cat = sat_cat.index[0] if len(sat_cat) else None
    if longest_cat:
        suggestions.append('“%s”评分最低，建议质检该品类物流时效与退换流程。' % longest_cat)
    bad_dev = c_dev.index[0] if len(c_dev) else None
    if bad_dev:
        suggestions.append('“%s”用户投诉集中，建议为其提供专属客服通道与优先受理。' % bad_dev)
    suggestions.append('针对高投诉客户（Top10）建立一对一跟进档案，降低流失风险。')
    suggestions.append('结合延迟订单推送物流异常主动提醒，前置化解服务投诉。')

    return {
        'kpis': {'avg_sat': _num(avg_sat), 'total_complaints': total_comp, 'comp_users': comp_users,
                 'comp_rate': _num(comp_rate, 2), 'neg_rate': _num(neg_rate, 2), 'avg_rating': _num(avg_rating),
                 'reviews': int(len(r))},
        'sentiment': [{'name': '好评', 'value': pos}, {'name': '中评', 'value': neu}, {'name': '差评', 'value': neg}],
        'by_device': [{'name': k, 'value': int(v)} for k, v in c_dev.items()],
        'by_city': [{'name': '%s线城市' % k, 'value': int(v)} for k, v in c_city.items()],
        'by_age': [{'name': str(k), 'value': int(v)} for k, v in c_age.items()],
        'neg_cat': [{'name': k, 'value': int(v)} for k, v in ncat.items()],
        'sat_cat': [{'name': k, 'value': _num(v, 2)} for k, v in sat_cat.items()],
        'combo': combo,
        'risk_users': [{'user_id': int(x['user_id']), 'sat': int(x['satisfaction']), 'comp': int(x['complaints']),
                        'device': x['device'], 'fav': x['fav_category'], 'risk': _num(x['risk'], 2)}
                       for _, x in top.iterrows()],
        'suggestions': suggestions,
    }



# ---------------- 外部公开数据集（自增数据源：UCI Online Retail）----------------
_EXTERNAL_SRC = 'https://archive.ics.uci.edu/ml/datasets/Online+Retail'
_UCI_XLSX_URL = 'https://archive.ics.uci.edu/ml/machine-learning-databases/00352/Online%20Retail.xlsx'


def _ensure_online_retail():
    """确保外部数据源文件存在；缺损时自动从 UCI 下载 xlsx（公网部署环境无该文件时使用）。"""
    gz = os.path.join(DATA, '_online_retail.csv.gz')
    xls = os.path.join(DATA, '_online_retail.xlsx')
    if os.path.exists(gz) or os.path.exists(xls):
        return
    os.makedirs(DATA, exist_ok=True)
    print('[数据] 本地缺少外部数据集，正在从 UCI 下载（约 23MB）...', flush=True)
    try:
        urllib.request.urlretrieve(_UCI_XLSX_URL, xls)
    except Exception as e:
        raise RuntimeError('外部数据集自动下载失败（%s）。请将 _online_retail.xlsx 手动放入 data/ 目录。' % e)
    print('[数据] 外部数据集下载完成。', flush=True)


def _load_online_retail():
    """加载外部数据集：优先用压缩 CSV（秒级读取），否则回退到 xlsx，并缓存结果。

    原实现直接用 `pd.read_excel` 读 23MB 的 xlsx，首次访问「外部数据源」页需约 30~40 秒，
    是页面跳转缓慢的最主要瓶颈。改用 `_online_retail.csv.gz`（约 7.6MB，读取 0.5s 内）后，
    首次打开该页即可秒开。
    """
    if '_or' not in _cache:
        gz = os.path.join(DATA, '_online_retail.csv.gz')
        xls = os.path.join(DATA, '_online_retail.xlsx')
        if os.path.exists(gz):
            _cache['_or'] = pd.read_csv(gz, compression='gzip')
        else:
            print('[数据] 暂无压缩缓存，正在读取 xlsx（首次较慢，约需数十秒）...', flush=True)
            _cache['_or'] = pd.read_excel(xls)
    return _cache['_or']


_ext_cache = None  # 外部数据集分析结果缓存（静态数据，计算结果可复用）


def external_data():
    """加载外部公开电商数据集 UCI Online Retail，产出全球市场洞察。"""
    global _ext_cache
    if _ext_cache is not None:
        return _ext_cache
    _ensure_online_retail()
    df = _load_online_retail().copy()
    df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'])
    df['month'] = df['InvoiceDate'].dt.to_period('M').astype(str)
    df['weekday'] = df['InvoiceDate'].dt.strftime('%A')

    # 有效交易：数量>0 且 单价>0
    d = df[(df['Quantity'] > 0) & (df['UnitPrice'] > 0)].copy()
    d['amt'] = d['Quantity'] * d['UnitPrice']

    total_trans = int(d['InvoiceNo'].nunique())
    total_rev = float(d['amt'].sum())
    avg_order = total_rev / total_trans
    n_products = int(d['StockCode'].nunique())
    n_users = int(d['CustomerID'].nunique())
    returns = df[df['Quantity'] < 0]
    ret_rate = len(returns) / len(df) * 100

    monthly = d.groupby('month')['amt'].sum().sort_index()
    country = d.groupby('Country')['amt'].sum().sort_values(ascending=False).head(10)
    prod = d.groupby('Description')['amt'].sum().sort_values(ascending=False).head(10)
    cust = d.groupby('CustomerID')['amt'].sum().sort_values(ascending=False).head(10)

    wk = d.groupby('weekday')['amt'].sum()
    order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    weekday = [{'name': w, 'value': _num(wk.get(w, 0))} for w in order]

    _ext_cache = {
        'source': _EXTERNAL_SRC,
        'desc': '本模块数据来源于 UCI 机器学习库公开数据集「Online Retail」(2010-12 至 2011-12，英国零售电商，'
                '541,909 条交易记录)，在赛题指定的阿里天池 Rec-Tmall 等基础数据之外，作为自增公开数据源，'
                '用于补充全球市场与生命周期洞察。',
        'kpis': {'transactions': total_trans, 'revenue': _num(total_rev), 'avg_order': _num(avg_order),
                 'products': n_products, 'customers': n_users, 'countries': int(df['Country'].nunique()),
                 'period': str(df['month'].min()) + ' ~ ' + str(df['month'].max())},
        'ret_rate': _num(ret_rate, 2),
        'monthly': [{'name': m, 'value': _num(v)} for m, v in monthly.items()],
        'country': [{'name': k, 'value': _num(v)} for k, v in country.items()],
        'products': [{'name': (k[:28] + '…') if len(k) > 28 else k, 'value': _num(v)} for k, v in prod.items()],
        'weekday': weekday,
        'customers': [{'id': int(k), 'value': _num(v)} for k, v in cust.items()],
    }
    return _ext_cache

