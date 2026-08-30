# -*- coding: utf-8 -*-
"""
数据生成脚本：以《项目2：直播电商用户画像》的 4544 个真实电商用户为基座，
生成配套的商品、浏览行为、订单、物流、评价明细数据，构成完整可用于
"电子商务大数据分析与智能决策支持系统" 的数据集。
输出到 data/ 目录下的 CSV 文件。
"""
import os
import numpy as np
import pandas as pd

BASE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE, 'data')
os.makedirs(DATA_DIR, exist_ok=True)

rng = np.random.default_rng(42)

CATEGORIES = ['Laptop & Accessory', 'Mobile Phone', 'Household', 'Fashion', 'Others', 'Grocery']

CAT_PRICE = {'Laptop & Accessory': (299, 8999), 'Mobile Phone': (999, 7999),
             'Household': (49, 899), 'Fashion': (89, 599),
             'Others': (39, 599), 'Grocery': (12, 189)}


def load_users():
    """从真实 xlsx 读取用户画像；失败则生成结构化用户。"""
    src = r'D:\cxdownload\项目2：直播电商用户画像.xlsx'
    try:
        df = pd.read_excel(src, sheet_name='E Comm')
    except Exception as e:
        print('读取真实用户失败，使用生成用户：', e)
        n = 4544
        df = pd.DataFrame({
            '客户ID': np.arange(50000, 50000 + n),
            '使用平台的月数': rng.integers(1, 30, n),
            '首选登录设备': rng.choice(['Mobile Phone', 'PC', 'Tablet'], n),
            '城市等级': rng.integers(1, 5, n),
            '仓库到家的距离': rng.integers(1, 60, n),
            '婚姻状况': rng.choice(['Married', 'Single'], n),
            '年龄组': rng.integers(1, 6, n),
            '性别': rng.choice(['Male', 'Female'], n),
            '在应用上花费的小时数': rng.integers(1, 20, n),
            '上个月下单的总次数': rng.integers(1, 30, n),
            '自上次订单以来的天数': rng.integers(1, 30, n),
            '首选订单类别': rng.choice(CATEGORIES, n),
            '关注的主播数量': rng.integers(0, 30, n),
            '满意度评分': rng.integers(1, 6, n),
            '投诉次数': rng.integers(0, 5, n),
            '上个月使用的优惠券总数': rng.integers(0, 15, n),
        })
    df = df.dropna(subset=['客户ID']).reset_index(drop=True)
    df['user_id'] = np.arange(1, len(df) + 1)
    df = df.rename(columns={
        '客户ID': 'customer_id', '使用平台的月数': 'months_used', '首选登录设备': 'device',
        '城市等级': 'city_level', '仓库到家的距离': 'warehouse_dist', '婚姻状况': 'marital',
        '年龄组': 'age_group', '性别': 'gender', '在应用上花费的小时数': 'hours_app',
        '上个月下单的总次数': 'orders_last_month', '自上次订单以来的天数': 'days_since_order',
        '首选订单类别': 'fav_category', '关注的主播数量': 'followed_anchor',
        '满意度评分': 'satisfaction', '投诉次数': 'complaints',
        '上个月使用的优惠券总数': 'coupons_used'})
    keep = ['user_id', 'customer_id', 'months_used', 'device', 'city_level', 'warehouse_dist',
            'marital', 'age_group', 'gender', 'hours_app', 'orders_last_month',
            'days_since_order', 'fav_category', 'followed_anchor', 'satisfaction',
            'complaints', 'coupons_used']
    df = df[keep]
    # 清理数值空值，避免下游计算 NaN
    num_cols = ['months_used', 'city_level', 'warehouse_dist', 'age_group', 'hours_app',
                'orders_last_month', 'days_since_order', 'followed_anchor', 'satisfaction',
                'complaints', 'coupons_used']
    for c in num_cols:
        df[c] = pd.to_numeric(df[c], errors='coerce')
        med = df[c].median()
        df[c] = df[c].fillna(med if pd.notna(med) else 0).astype(int)
    return df


def gen_products():
    rows = []
    pid = 0
    for cat in CATEGORIES:
        lo, hi = CAT_PRICE[cat]
        n = 12  # 每类商品数
        for i in range(n):
            pid += 1
            price = int(round(rng.uniform(lo, hi), -1))
            cost = int(round(price * rng.uniform(0.45, 0.68), -1))
            rows.append({
                'product_id': f'P{pid:04d}', 'name': f'{cat}商品{i+1:02d}', 'category': cat,
                'price': price, 'cost': cost,
                'stock': int(rng.integers(50, 2000)),
                'launch_date': pd.Timestamp('2024-01-01') + pd.Timedelta(days=int(rng.integers(0, 900))),
            })
    return pd.DataFrame(rows)


def gen_orders(users, products):
    """生成订单：用户活跃度、类别偏好、季节性、促销、价格弹性影响。"""
    n = 26000
    pids_all = products['product_id'].values
    # 用户活跃度权重
    users['_act'] = 1 + users['hours_app'] * 0.5 + users['orders_last_month']
    w = users['_act'].values / users['_act'].values.sum()
    dates = pd.date_range('2025-08-01', '2026-08-29', freq='D')

    def seasonal(t):
        d = t.dayofweek
        s = 1.0
        if d >= 5:
            s *= 1.5
        if t.month == 11 and 1 <= t.day <= 15:
            s *= 4.0
        if t.month == 6 and 10 <= t.day <= 20:
            s *= 3.0
        if d == 0:
            s *= 1.2
        return s

    daily_probs = np.array([seasonal(x) for x in dates])
    daily_probs = daily_probs / daily_probs.sum()
    order_days = rng.choice(len(dates), size=n, p=daily_probs)
    order_dates = dates[order_days]

    user_idx = rng.choice(len(users), size=n, p=w)
    user_ids = users['user_id'].values[user_idx]
    u_cat = users['fav_category'].values[user_idx]
    # 商品按类别分组，并按价格反比加权（价格越低越畅销，形成价格弹性信号）
    all_prods = products.copy()
    all_prods['_w'] = (1.0 / all_prods['price'].astype(float)) ** 0.8
    prods_by_cat = {c: g['product_id'].tolist() for c, g in all_prods.groupby('category')}
    w_by_pid = dict(zip(all_prods['product_id'], all_prods['_w']))

    def pick_pid(cat):
        cands = prods_by_cat.get(cat)
        if not cands:
            cands = list(prods_by_cat.values())[0]
        ws = np.array([w_by_pid[p] for p in cands])
        ws = ws / ws.sum()
        return cands[int(rng.choice(len(cands), p=ws))]

    rows = []
    for i in range(n):
        cat = u_cat[i]
        if rng.random() < 0.7 and cat in prods_by_cat:
            pid = pick_pid(cat)
        else:
            pid = pick_pid(rng.choice(list(prods_by_cat.keys())))
        prod = products[products['product_id'] == pid].iloc[0]
        base_price = int(prod['price'])
        disc = rng.choice([1.0, 0.9, 0.8, 0.7, 0.6], p=[0.3, 0.3, 0.2, 0.15, 0.05])
        unit_price = round(base_price * disc, 2)
        d = order_dates[i]
        if d.month == 11 or (d.month == 6 and 10 <= d.day <= 20):
            unit_price = round(unit_price * 0.9, 2)
        # 需求负相关：成交单价越低，单次购买数量越多（价格弹性信号）
        qty = max(1, int(round(3.2 - unit_price / 1200.0 + rng.normal(0, 0.4))))
        channel = rng.choice(['App', 'Web', 'MiniProgram'], p=[0.6, 0.25, 0.15])
        rows.append({
            'order_id': f'ORD{100000+i:05d}', 'user_id': user_ids[i], 'product_id': pid,
            'category': prod['category'], 'qty': qty, 'unit_price': unit_price,
            'paid_amount': round(unit_price * qty, 2), 'discount': round(1 - unit_price / base_price, 3),
            'order_date': d, 'channel': channel,
        })
    odf = pd.DataFrame(rows)
    odf['order_date'] = pd.to_datetime(odf['order_date'])
    odf = odf.merge(users[['user_id', 'warehouse_dist', 'city_level']], on='user_id', how='left')
    return odf


def gen_behaviors(users, products):
    """浏览/加购等行为，构建推荐用交互矩阵。"""
    n = 42000
    users['_act'] = 1 + users['hours_app'] * 0.8 + users['orders_last_month'] * 0.3
    w = users['_act'].values / users['_act'].values.sum()
    uid = users['user_id'].values[rng.choice(len(users), size=n, p=w)]
    pids = products['product_id'].values
    type_ = rng.choice(['view', 'cart', 'purchase'], size=n, p=[0.7, 0.2, 0.1])
    sel = rng.choice(len(pids), size=n)
    return pd.DataFrame({'user_id': uid, 'product_id': pids[sel], 'behavior_type': type_,
                         'timestamp': pd.date_range('2026-07-01', periods=n, freq='min')})


def gen_logistics(orders, users):
    """物流：送达时间与仓库距离、类别、城市等级相关。"""
    rows = []
    for _, o in orders.iterrows():
        dist = o['warehouse_dist'] if pd.notna(o.get('warehouse_dist')) else 15
        cl = o['city_level'] if pd.notna(o.get('city_level')) else 2
        base = int(rng.normal(2 + dist / 12.0 + cl * 0.3, 0.8))
        ship = max(0, int(rng.normal(0.5, 0.5)))
        arrival = max(0, int(np.round(base)))
        status = 'delivered' if arrival <= 5 else 'delayed'
        rows.append({'order_id': o['order_id'], 'ship_date': o['order_date'] + pd.Timedelta(days=ship),
                     'arrival_date': o['order_date'] + pd.Timedelta(days=arrival),
                     'delivery_days': arrival, 'status': status})
    return pd.DataFrame(rows)


def gen_reviews(orders, users):
    """评价：满意度与送达、投诉相关。"""
    s = orders.sample(min(9000, len(orders)), random_state=7)
    ratings = []
    for _, o in s.iterrows():
        base = 4.0
        if o.get('delivery_days', 3) > 5:
            base -= 1.5
        cl = o['city_level'] if pd.notna(o.get('city_level')) else 2
        base += cl / 5.0 - 0.2
        r = int(np.clip(round(base + rng.normal(0, 0.6)), 1, 5))
        ratings.append(r)
    s = s.copy()
    s['rating'] = ratings
    s['sentiment'] = np.where(s['rating'] >= 4, 'positive', np.where(s['rating'] == 3, 'neutral', 'negative'))
    s['review_date'] = s['order_date'] + pd.Timedelta(days=3)
    return s[['order_id', 'user_id', 'product_id', 'category', 'rating', 'sentiment', 'review_date']]


def main():
    print('生成用户...')
    users = load_users()
    products = gen_products()
    print('生成订单...')
    orders = gen_orders(users, products)
    print('生成行为...')
    behaviors = gen_behaviors(users, products)
    print('生成物流...')
    logistics = gen_logistics(orders, users)
    orders = orders.merge(logistics[['order_id', 'delivery_days', 'status']], on='order_id', how='left')
    print('生成评价...')
    reviews = gen_reviews(orders, users)

    users[['user_id', 'customer_id', 'months_used', 'device', 'city_level', 'warehouse_dist',
           'marital', 'age_group', 'gender', 'hours_app', 'orders_last_month', 'days_since_order',
           'fav_category', 'followed_anchor', 'satisfaction', 'complaints', 'coupons_used']].to_csv(
        os.path.join(DATA_DIR, 'users.csv'), index=False, encoding='utf-8-sig')
    products.to_csv(os.path.join(DATA_DIR, 'products.csv'), index=False, encoding='utf-8-sig')
    orders.to_csv(os.path.join(DATA_DIR, 'orders.csv'), index=False, encoding='utf-8-sig')
    behaviors.to_csv(os.path.join(DATA_DIR, 'behaviors.csv'), index=False, encoding='utf-8-sig')
    logistics.to_csv(os.path.join(DATA_DIR, 'logistics.csv'), index=False, encoding='utf-8-sig')
    reviews.to_csv(os.path.join(DATA_DIR, 'reviews.csv'), index=False, encoding='utf-8-sig')

    print(f"完成：users={len(users)}, products={len(products)}, orders={len(orders)}, "
          f"behaviors={len(behaviors)}, logistics={len(logistics)}, reviews={len(reviews)}")


if __name__ == '__main__':
    main()
