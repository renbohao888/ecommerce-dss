# -*- coding: utf-8 -*-
"""AI 数据问答助手：为可拖动悬浮窗提供数据问答能力。"""
from ml import models as M


def _ai_line(x, y, name='销售额'):
    return {'grid': {'left': 44, 'right': 18, 'top': 26, 'bottom': 30},
            'tooltip': {'trigger': 'axis'},
            'xAxis': {'type': 'category', 'data': x, 'boundaryGap': False},
            'yAxis': {'type': 'value'},
            'color': ['#22d3ee'],
            'series': [{'type': 'line', 'name': name, 'smooth': True, 'symbol': 'circle',
                        'symbolSize': 6, 'data': y, 'lineStyle': {'width': 3, 'shadowBlur': 14,
                                                                  'shadowColor': 'rgba(34,211,238,.55)'},
                        'areaStyle': {'opacity': .35}}]}


def _ai_bar(x, y, name='金额'):
    return {'grid': {'left': 50, 'right': 18, 'top': 26, 'bottom': 38},
            'tooltip': {'trigger': 'axis'},
            'xAxis': {'type': 'category', 'data': x, 'axisLabel': {'rotate': 26}},
            'yAxis': {'type': 'value'},
            'color': ['#3b82f6'],
            'series': [{'type': 'bar', 'name': name, 'data': y, 'barMaxWidth': 34,
                        'itemStyle': {'borderRadius': [6, 6, 0, 0], 'shadowBlur': 12,
                                      'shadowColor': 'rgba(59,130,246,.45)'}}]}


def _ai_pie(data):
    return {'tooltip': {'trigger': 'item'}, 'legend': {'bottom': 0},
            'color': ['#f43f5e', '#fbbf24', '#3b82f6', '#10b981'],
            'series': [{'type': 'pie', 'radius': ['40%', '62%'], 'center': ['50%', '42%'],
                        'data': data, 'label': {'color': '#cbd5e1'},
                        'itemStyle': {'borderColor': '#0b1020', 'borderWidth': 2}}]}


def ai_answer(question):
    """按问题意图返回 {'reply','chart','suggestions'}。"""
    q = (question or '').strip()
    low = q.lower()

    def out(reply, chart=None, suggestions=None):
        return {'reply': reply, 'chart': chart, 'suggestions': suggestions or []}

    def fm(v):
        return '¥{:,.0f}'.format(v) if v is not None else '—'

    def overview():
        return ('📊 平台经营概览：累计商品交易额 **%s**，订单 **%s** 单，在售商品 **%s** 个，'
                '注册用户 **%s** 人，客单价 **%s**。\n'
                '· 平均满意度评分 **%.2f**（满分 5）\n'
                '· 投诉用户 **%s** 人，物流延迟率 **%.2f%%**\n'
                '· 评价总数 **%s** 条'
                % (fm(k['gmv']), k['orders'], k['products'], k['users'], fm(k['avg_order']),
                   k['avg_sat'], k['complaints'], k['delay_rate'], k['reviews']))

    try:
        dash = M.dashboard_data()
    except Exception as e:
        return out('抱歉，数据加载出现异常：%s' % e, suggestions=['看一下数据总览', '热销商品有哪些'])
    k = dash['kpis']
    SUG = ['总销售额', '热销品类排行榜', '每天销售走势', '物流延迟率是多少', '预测未来14天销量',
           '客户分层与营销建议', '哪些品类容易差评', '品类最优定价', '外部数据集有什么']

    # 问候
    if any(w in low for w in ['你好', '您好', 'hi', 'hello', '嗨', '在吗']):
        return out('你好！我是系统内置的 **AI 数据助手** 🤖。我可以回答平台各模块的数据问题，'
                   '例如总销售额、热销品类、销售走势、物流延迟率、预测未来销量、客户分层、最优定价、外部数据集等。\n'
                   '直接输入问题或点击下方推荐问题即可。', suggestions=SUG)
    # 帮助
    if any(w in low for w in ['帮助', '能做什么', '功能', '会什么', '模块', '有哪些', '怎么用']):
        return out('我能回答以下内容：\n'
                   '· **经营总览**：总销售额、订单量、客单价、满意度、投诉、延迟率\n'
                   '· **销售分析**：每日销售走势、热销品类、渠道、爆款商品、用户画像\n'
                   '· **推荐与预测**：推荐算法逻辑、未来销量预测\n'
                   '· **营销与客户**：RFM 客户分层与营销建议、客户满意度与差评归因、风险客户\n'
                   '· **供应链**：物流延迟率与风险因素\n'
                   '· **定价**：价格弹性与最优定价\n'
                   '· **外部数据**：全球市场洞察、退货率', suggestions=SUG)
    # 预测（优先，避免被“销量/商品”截胡）
    if any(w in low for w in ['预测', '未来', '销量预测', '下月', '预估']):
        f = M.forecast(14)
        fut = f['dates']
        tot = float(sum(f['forecast']))
        return out('🔮 基于随机森林时序回归，未来 **%s** 天（%s ~ %s）预计销售额 **%s**，日均约 **%s**。'
                   '模型 MAE=**%s**。' % (len(fut), fut[0], fut[-1], fm(tot), fm(tot / len(fut)), f['mae']),
                   _ai_line(fut, f['forecast'], '预测销售额'), ['总销售额', '每天销售走势', '热销品类'])
    # 推荐
    if any(w in low for w in ['推荐', '猜你喜欢', '个性化', '协同过滤']):
        return out('🧠 **推荐模块**采用「基于物品的协同过滤」（物品-用户矩阵 + 余弦相似度）：\n'
                   '· 老用户：推荐与 TA 历史交互商品相似度最高的商品\n'
                   '· 新用户：推荐平台热销商品兜底\n'
                   '可在「用户行为与推荐」页输入用户 ID 查看个性化推荐。', None,
                   ['热销商品有哪些', '总销售额', '预测未来销量'])
    # 总销售额/营收
    if any(w in low for w in ['总销售额', '交易额', '营收', 'gmv', '总额', '卖了多少钱', '销售额']):
        d = dash['daily']
        return out(overview(), _ai_line(d['date'], d['gmv'], '销售额'), SUG)
    # 客单价
    if any(w in low for w in ['客单价', '平均订单', '平均每单', '人均']):
        return out('💳 平台客单价 **%s**（商品交易额 %s ÷ 订单 %s 单）。'
                   % (fm(k['avg_order']), fm(k['gmv']), k['orders']), None, ['总销售额', '订单量', '每天销售走势'])
    # 订单/销量
    if any(w in low for w in ['订单量', '订单数', '成交量', '多少单', '订单', '销量']):
        d = dash['daily']
        return out('🧾 平台累计订单 **%s** 单，商品交易额 **%s**，客单价 **%s**。\n'
                   '近 30 天共 **%s** 单，日均约 **%.0f** 单。'
                   % (k['orders'], fm(k['gmv']), fm(k['avg_order']),
                      int(sum(d['orders'])), float(sum(d['orders'])) / max(1, len(d['orders']))),
                   _ai_line(d['date'], d['orders'], '订单数'), ['总销售额', '每天销售走势', '预测未来销量'])
    # 满意度/评分/好评
    if any(w in low for w in ['满意度', '评分', '好评', '评价', '满意']):
        pie = [{'name': '%s星' % s['rating'], 'value': s['count']} for s in dash['satisfaction']]
        return out('⭐ 平均满意度评分 **%.2f**（满分 5），共 **%s** 条评价，星级分布如下。'
                   % (k['avg_sat'], k['reviews']), _ai_pie(pie), ['哪些品类容易差评', '总销售额', '客户分层与营销建议'])
    # 投诉
    if any(w in low for w in ['投诉', '客诉']):
        svc = M.customer_service()
        sp = svc['kpis']
        return out('⚠️ 投诉相关：投诉用户 **%s** 人（投诉率 **%.2f%%**），累计 **%s** 件，'
                   '平均满意度 **%.2f**，负面评价占比 **%.2f%%**。\n**服务建议**：\n%s'
                   % (sp['comp_users'], sp['comp_rate'], sp['total_complaints'], sp['avg_sat'],
                      sp['neg_rate'], '\n'.join('- ' + s for s in svc['suggestions'][:3])),
                   _ai_pie(svc['sentiment']), ['哪些品类容易差评', '风险客户有哪些', '客户分层与营销建议'])
    # 趋势/每日/近期
    if any(w in low for w in ['趋势', '走势', '每日', '每天', '近期', '最近', '近30', '销售曲线']):
        d = dash['daily']
        return out('📈 这是近 30 天每日销售走势，最近一天（%s）成交 **%s** 单、销售额 **%s**。'
                   % (d['date'][-1], d['orders'][-1], fm(d['gmv'][-1])),
                   _ai_line(d['date'], d['gmv'], '销售额'), SUG)
    # 客服/差评/风险客户（须在“客户/品类”之前）
    if any(w in low for w in ['客服', '差评', '负面', '风险客户', '售后', '客户服务']):
        svc = M.customer_service()
        sp = svc['kpis']
        return out('🎧 **客户服务洞察**：平均满意度 **%.2f**，负面评价率 **%.2f%%**，投诉用户 **%s** 人。\n'
                   '**优化建议**：\n%s'
                   % (sp['avg_sat'], sp['neg_rate'], sp['comp_users'],
                      '\n'.join('- ' + s for s in svc['suggestions'][:4])),
                   _ai_pie(svc['sentiment']), ['哪些品类容易差评', '风险客户有哪些', '客户分层与营销建议'])

    # 营销/RFM/分层（须在“客户”之前）
    if any(w in low for w in ['营销', 'rfm', '分层', '分群', '流失', '高价值', '活跃客户']):
        mk = M.marketing_data()
        segs = mk['segs']
        return out('🎯 基于 RFM（最近购买/频次/金额）+ K-Means 将客户分为 4 类：\n%s'
                   % '\n'.join('· **%s**：%s 人（%.1f%%），策略：%s' % (s['name'], s['count'], s['pct'], s['strategy'])
                               for s in segs),
                   _ai_bar([s['name'] for s in segs], [s['count'] for s in segs], '人数'),
                   ['客户分层与营销建议', '风险客户有哪些', '总销售额'])
    # 供应链/物流
    if any(w in low for w in ['供应链', '物流', '延迟', '配送', '时效', '仓库', '风控']):
        sp = M.supplychain_data()
        imp = sp['imp'][:6]
        return out('🚚 当前物流 **延迟率 %.2f%%**。\n主要延迟影响因素（随机森林特征重要性）：\n%s'
                   % (sp['delay_rate'],
                      '\n'.join('· %s（影响度 %.1f%%）' % (i['name'], i['value'] * 100) for i in imp)),
                   _ai_bar([d['name'] for d in sp['by_dist']], [d['value'] for d in sp['by_dist']], '延迟率'),
                   ['哪些品类延迟高', '物流延迟率是多少', '总销售额'])
    # 价格/定价/调价
    if any(w in low for w in ['价格', '定价', '弹性', '调价', '最优价格', '涨价', '降价']):
        pd_ = M.price_data()
        rows = pd_['rows'][:8]
        return out('💰 已为 **%s** 个商品完成价格弹性估计与最优定价。\n'
                   '定价思路：弹性 < -1 用垄断最优价，-1 ~ -0.5 小幅降价扩量，> -0.5 适当提价。\n'
                   '调整后收入变化最大的商品如下。' % pd_['count'],
                   _ai_bar([r['name'][:6] for r in rows], [r['delta_rev'] for r in rows], '收入增量'),
                   ['品类最优定价', '热销品类', '总销售额'])
    # 渠道
    if any(w in low for w in ['渠道', '流量', '来源', '抖音', '直播', '短视频', '平台']):
        ch = dash['channels'][:8]
        return out('📡 各流量渠道销售分布如下，最强势渠道为 **%s**。'
                   % (ch[0]['name'] if ch else '—'),
                   _ai_bar([c['name'] for c in ch], [c['value'] for c in ch], '销售额'),
                   ['热销品类', '总销售额'])
    # 品类/类目/热销品类
    if any(w in low for w in ['品类', '类目', '分类', '热销品类', '哪个品类', '卖得好']):
        cs = dash['cat_sales'][:8]
        top = cs[0] if cs else None
        return out('🛍️ 销售最好的品类是 **「%s」**（%s）。各品类销售金额如下。'
                   % (top['name'], fm(top['value']) if top else '—'),
                   _ai_bar([c['name'] for c in cs], [c['value'] for c in cs], '销售额'),
                   ['渠道分布', '热销商品有哪些', '总销售额'])
    # 热销商品
    if any(w in low for w in ['商品', '爆款', '热销商品', '卖得最好', '排行', 'top']):
        tp = dash['top_prods'][:6]
        return out('🏆 平台热销 TOP 商品：\n%s'
                   % '\n'.join('%d. **%s**（%s）—— %s' % (i + 1, t['name'], t['category'], fm(t['value']))
                               for i, t in enumerate(tp)), None, ['总销售额', '热销品类', '预测未来销量'])
    # 外部数据集
    if any(w in low for w in ['外部', '数据源', '全球', '海外', '零售', 'online', '退货', '退款']):
        ext = M.external_data()
        kp = ext['kpis']
        return out('🌐 **外部公开数据集**（UCI Online Retail，%s，%s 条交易）：\n'
                   '· 交易笔数 **%s**，营收 **%s**，客单价 **%s**\n'
                   '· 商品 **%s** 个，客户 **%s** 位，覆盖 **%s** 个国家\n'
                   '· 退货率 **%.2f%%**'
                   % (kp['period'], kp['transactions'], kp['transactions'], fm(kp['revenue']),
                      fm(kp['avg_order']), kp['products'], kp['customers'], kp['countries'], ext['ret_rate']),
                   _ai_bar([m['name'] for m in ext['monthly'][:12]],
                           [m['value'] for m in ext['monthly'][:12]], '月度营收'),
                   ['外部数据集有什么', '总销售额', '每天销售走势'])
    # 用户/客户/会员（通用，最后匹配）
    if any(w in low for w in ['用户', '客户', '会员', '人群', '消费者']):
        u = M.users()
        gen = u['gender'].value_counts()
        gmap = {'male': '男', 'female': '女'}
        return out('👥 平台注册用户 **%s** 人，在售商品 **%s** 个。\n性别分布：%s\n平均满意度 **%.2f**，投诉用户 **%s** 人。'
                   % (k['users'], k['products'],
                      '，'.join('%s %s 人' % (gmap.get(i, i), int(v)) for i, v in gen.items()) or '—',
                      k['avg_sat'], k['complaints']),
                   _ai_pie([{'name': gmap.get(i, i), 'value': int(v)} for i, v in gen.items()]),
                   ['总销售额', '用户满意度如何', '客户分层与营销建议'])
    # 兜底：概览
    return out(overview(), _ai_line(dash['daily']['date'], dash['daily']['gmv'], '销售额'), SUG)

