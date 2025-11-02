import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Tuple
import plotly.graph_objects as go
import plotly.express as px
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.spatial.distance import squareform
from sklearn.covariance import LedoitWolf

# 페이지 설정
st.set_page_config(
    page_title="고급 원자재 포트폴리오 최적화",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS 스타일링
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 1rem;
    }
    .regime-bullish {
        background: linear-gradient(135deg, #00aa00 0%, #28a745 100%);
        color: white;
        padding: 1.5rem;
        border-radius: 12px;
        text-align: center;
        box-shadow: 0 4px 6px rgba(0,0,0,0.2);
    }
    .regime-bearish {
        background: linear-gradient(135deg, #dc3545 0%, #990000 100%);
        color: white;
        padding: 1.5rem;
        border-radius: 12px;
        text-align: center;
        box-shadow: 0 4px 6px rgba(0,0,0,0.2);
    }
    .regime-neutral {
        background: linear-gradient(135deg, #ffc107 0%, #ff9800 100%);
        color: white;
        padding: 1.5rem;
        border-radius: 12px;
        text-align: center;
        box-shadow: 0 4px 6px rgba(0,0,0,0.2);
    }
    .metric-card {
        background: white;
        padding: 1rem;
        border-radius: 8px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        border-left: 4px solid #667eea;
    }
    .warning-box {
        background: #fff3cd;
        border-left: 4px solid #ffc107;
        padding: 1rem;
        border-radius: 4px;
        margin: 1rem 0;
    }
    .info-box {
        background: #d1ecf1;
        border-left: 4px solid #17a2b8;
        padding: 1rem;
        border-radius: 4px;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# 원자재 ETF 설정
COMMODITY_ETFS = {
    # 금속
    'gold': {'symbol': 'GLD', 'name': '금 (Gold)', 'category': 'precious_metals', 'color': '#FFD700'},
    'silver': {'symbol': 'SLV', 'name': '은 (Silver)', 'category': 'precious_metals', 'color': '#C0C0C0'},
    'copper': {'symbol': 'COPX', 'name': '구리 (Copper Miners)', 'category': 'industrial_metals', 'color': '#B87333'},
    'copper_futures': {'symbol': 'CPER', 'name': '구리 선물 (Copper Futures)', 'category': 'industrial_metals', 'color': '#CD7F32'},
    
    # 에너지
    'oil_uso': {'symbol': 'USO', 'name': '원유 USO (Oil)', 'category': 'energy', 'color': '#000000'},
    'oil_dbo': {'symbol': 'DBO', 'name': '원유 DBO (최적화)', 'category': 'energy', 'color': '#1a1a1a'},
    'natural_gas': {'symbol': 'UNG', 'name': '천연가스 (Natural Gas)', 'category': 'energy', 'color': '#4169E1'},
    
    # 희귀원소
    'rare_earth': {'symbol': 'REMX', 'name': '희귀원소 (Rare Earth)', 'category': 'strategic', 'color': '#8B008B'},
    
    # 농산물
    'corn': {'symbol': 'CORN', 'name': '옥수수 (Corn)', 'category': 'agriculture', 'color': '#FFD700'},
    'wheat': {'symbol': 'WEAT', 'name': '밀 (Wheat)', 'category': 'agriculture', 'color': '#DEB887'},
    'soybean': {'symbol': 'SOYB', 'name': '대두 (Soybean)', 'category': 'agriculture', 'color': '#8B4513'},
    
    # 광범위 원자재
    'dbc': {'symbol': 'DBC', 'name': 'Invesco DB Commodity', 'category': 'broad', 'color': '#2F4F4F'},
    'gsg': {'symbol': 'GSG', 'name': 'iShares S&P GSCI', 'category': 'broad', 'color': '#556B2F'}
}

# 거시경제 지표
MACRO_INDICATORS = {
    'dxy': {'symbol': 'DX-Y.NYB', 'name': 'DXY (달러지수)'},
    'us10y': {'symbol': '^TNX', 'name': '미 10년물 국채'},
    'tips': {'symbol': 'TIP', 'name': 'TIPS (인플레이션 연동채)'},
    'spx': {'symbol': '^GSPC', 'name': 'S&P 500'},
    'vix': {'symbol': '^VIX', 'name': 'VIX (변동성)'}
}

# ============================================================================
# 데이터 수집
# ============================================================================

@st.cache_data(ttl=300)
def fetch_commodity_data(lookback_days=365):
    """원자재 데이터 수집"""
    data = {}
    end_date = datetime.now()
    start_date = end_date - timedelta(days=lookback_days + 60)  # MA 계산 여유
    
    for key, info in COMMODITY_ETFS.items():
        try:
            ticker = yf.Ticker(info['symbol'])
            hist = ticker.history(start=start_date, end=end_date)
            
            if not hist.empty:
                data[key] = {
                    'history': hist,
                    'current': hist['Close'].iloc[-1],
                    'prev': hist['Close'].iloc[-2] if len(hist) >= 2 else hist['Close'].iloc[-1],
                    'info': info
                }
        except Exception as e:
            st.warning(f"{info['name']} 데이터 로드 실패: {str(e)}")
            continue
    
    return data

@st.cache_data(ttl=300)
def fetch_macro_data():
    """거시경제 지표 수집"""
    data = {}
    
    for key, info in MACRO_INDICATORS.items():
        try:
            ticker = yf.Ticker(info['symbol'])
            hist = ticker.history(period="6mo")  # MA 계산용
            
            if not hist.empty:
                # 이동평균선 계산
                close_prices = hist['Close']
                ma50 = close_prices.rolling(window=50).mean()
                ma200 = close_prices.rolling(window=200).mean()
                
                # RSI 계산
                rsi = calculate_rsi(close_prices, 14)
                
                data[key] = {
                    'history': hist,
                    'current': close_prices.iloc[-1],
                    'prev': close_prices.iloc[-2] if len(close_prices) >= 2 else close_prices.iloc[-1],
                    'ma50': ma50.iloc[-1] if len(ma50) >= 50 else None,
                    'ma200': ma200.iloc[-1] if len(ma200) >= 200 else None,
                    'rsi': rsi.iloc[-1] if len(rsi) >= 14 else None,
                    'change_pct': ((close_prices.iloc[-1] - close_prices.iloc[-2]) / close_prices.iloc[-2] * 100) if len(close_prices) >= 2 else 0
                }
        except Exception as e:
            continue
    
    return data

def calculate_rsi(prices, period=14):
    """RSI 계산"""
    delta = prices.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

# ============================================================================
# DXY 필터 분석
# ============================================================================

def analyze_dxy_regime(macro_data):
    """DXY 레짐 분석"""
    if 'dxy' not in macro_data:
        return None
    
    dxy = macro_data['dxy']
    current = dxy['current']
    ma50 = dxy.get('ma50')
    ma200 = dxy.get('ma200')
    rsi = dxy.get('rsi')
    
    regime = {
        'current_price': current,
        'ma50': ma50,
        'ma200': ma200,
        'rsi': rsi,
        'signals': [],
        'score': 0,  # -2 (강력 약세) ~ +2 (강력 강세)
        'regime_type': 'neutral'
    }
    
    # 골든/데드 크로스 분석
    if ma50 and ma200:
        if ma50 > ma200:
            regime['signals'].append('🔴 골든 크로스 (달러 강세 레짐)')
            regime['score'] += 2
        else:
            regime['signals'].append('🟢 데드 크로스 (달러 약세 레짐)')
            regime['score'] -= 2
    
    # 가격 vs MA200
    if ma200:
        if current > ma200:
            regime['signals'].append('⚠️ 가격 > MA200 (장기 상승 추세)')
            regime['score'] += 1
        else:
            regime['signals'].append('✅ 가격 < MA200 (장기 하락 추세)')
            regime['score'] -= 1
    
    # RSI 과매수/과매도
    if rsi:
        if rsi > 70:
            regime['signals'].append(f'🟢 RSI 과매수 ({rsi:.1f}) - 반전 기대')
            regime['score'] -= 1  # 과매수는 하락 기대
        elif rsi < 30:
            regime['signals'].append(f'🔴 RSI 과매도 ({rsi:.1f}) - 반등 기대')
            regime['score'] += 1
        else:
            regime['signals'].append(f'🟡 RSI 중립 ({rsi:.1f})')
    
    # 최종 레짐 판단
    if regime['score'] >= 2:
        regime['regime_type'] = 'strong_bearish'  # 달러 강세 = 원자재 약세
        regime['regime_name'] = '🔴🔴 강력 방어 (달러 강세)'
        regime['recommendation'] = '원자재 비중 대폭 축소, 현금/국채 확대'
    elif regime['score'] >= 1:
        regime['regime_type'] = 'bearish'
        regime['regime_name'] = '🔴 방어 (달러 강세)'
        regime['recommendation'] = '원자재 비중 축소, 방어적 포지션'
    elif regime['score'] <= -2:
        regime['regime_type'] = 'strong_bullish'  # 달러 약세 = 원자재 강세
        regime['regime_name'] = '🟢🟢 강력 공격 (달러 약세)'
        regime['recommendation'] = '원자재 비중 대폭 확대, 성장 자산 집중'
    elif regime['score'] <= -1:
        regime['regime_type'] = 'bullish'
        regime['regime_name'] = '🟢 공격 (달러 약세)'
        regime['recommendation'] = '원자재 비중 확대, 공격적 포지션'
    else:
        regime['regime_type'] = 'neutral'
        regime['regime_name'] = '🟡 중립'
        regime['recommendation'] = '현 비중 유지, 관망'
    
    return regime

# ============================================================================
# HRP 최적화
# ============================================================================

def calculate_hrp_weights(returns_df):
    """계층적 위험 균형 (HRP) 가중치 계산"""
    
    # 1. 상관관계 행렬 계산
    corr_matrix = returns_df.corr()
    
    # 2. 거리 행렬 변환 (1 - correlation)
    dist_matrix = np.sqrt((1 - corr_matrix) / 2)
    
    # 3. 계층적 클러스터링
    linkage_matrix = linkage(squareform(dist_matrix.values), method='single')
    
    # 4. 클러스터 순서 정렬
    sorted_idx = _get_quasi_diag(linkage_matrix)
    sorted_corr = corr_matrix.iloc[sorted_idx, sorted_idx]
    
    # 5. 재귀적 이등분 (Recursive Bisection)
    weights = pd.Series(1.0, index=sorted_corr.index)
    clusters = [sorted_corr.columns.tolist()]
    
    while len(clusters) > 0:
        clusters = [cluster[start:end] for cluster in clusters
                   for start, end in ((0, len(cluster) // 2), (len(cluster) // 2, len(cluster)))
                   if len(cluster) > 1]
        
        for i in range(0, len(clusters), 2):
            if i + 1 < len(clusters):
                cluster0 = clusters[i]
                cluster1 = clusters[i + 1]
                
                # 클러스터 변동성 계산
                cov_matrix = returns_df[cluster0 + cluster1].cov()
                var0 = _get_cluster_var(cov_matrix, cluster0)
                var1 = _get_cluster_var(cov_matrix, cluster1)
                
                # 역변동성 가중치
                alpha = 1 - var0 / (var0 + var1)
                
                weights[cluster0] *= alpha
                weights[cluster1] *= (1 - alpha)
    
    return weights / weights.sum()

def _get_quasi_diag(linkage_matrix):
    """클러스터 트리에서 준대각 순서 추출"""
    link = linkage_matrix.astype(int)
    sort_idx = pd.Series([link[-1, 0], link[-1, 1]])
    num_items = link[-1, 3]
    
    while sort_idx.max() >= num_items:
        sort_idx.index = range(0, sort_idx.shape[0] * 2, 2)
        df0 = sort_idx[sort_idx >= num_items]
        i = df0.index
        j = df0.values - num_items
        sort_idx[i] = link[j, 0]
        df0 = pd.Series(link[j, 1], index=i + 1)
        sort_idx = pd.concat([sort_idx, df0])
        sort_idx = sort_idx.sort_index()
        sort_idx.index = range(sort_idx.shape[0])
    
    return sort_idx.tolist()

def _get_cluster_var(cov_matrix, cluster_items):
    """클러스터 분산 계산"""
    cov_slice = cov_matrix.loc[cluster_items, cluster_items]
    w = pd.Series(1 / len(cluster_items), index=cluster_items)
    return np.dot(w.T, np.dot(cov_slice, w))

# ============================================================================
# 비율 분석
# ============================================================================

def calculate_ratios(commodity_data):
    """주요 비율 계산"""
    ratios = {}
    
    # 금/은 비율
    if 'gold' in commodity_data and 'silver' in commodity_data:
        gold_price = commodity_data['gold']['current']
        silver_price = commodity_data['silver']['current']
        
        if silver_price > 0:
            gs_ratio = gold_price / silver_price
            
            if gs_ratio > 90:
                signal = '🟢🟢 은 강력매수'
                level = 'strong_buy_silver'
                desc = f'금은비율 {gs_ratio:.1f} - 은 심각한 저평가'
            elif gs_ratio > 82:
                signal = '🟢 은 매수'
                level = 'buy_silver'
                desc = f'금은비율 {gs_ratio:.1f} - 은 저평가'
            elif gs_ratio < 60:
                signal = '🔴🔴 금 강력매수'
                level = 'strong_buy_gold'
                desc = f'금은비율 {gs_ratio:.1f} - 금 심각한 저평가'
            elif gs_ratio < 68:
                signal = '🔴 금 매수'
                level = 'buy_gold'
                desc = f'금은비율 {gs_ratio:.1f} - 금 저평가'
            else:
                signal = '🟡 중립'
                level = 'neutral'
                desc = f'금은비율 {gs_ratio:.1f} - 정상 범위'
            
            ratios['gold_silver'] = {
                'ratio': gs_ratio,
                'signal': signal,
                'level': level,
                'description': desc
            }
    
    # 구리/금 비율 (경기 온도계)
    if 'copper' in commodity_data and 'gold' in commodity_data:
        # 선물 가격 사용 (더 정확)
        copper_hist = commodity_data['copper']['history']
        gold_hist = commodity_data['gold']['history']
        
        # 최근 가격
        copper_price = copper_hist['Close'].iloc[-1]
        gold_price = gold_hist['Close'].iloc[-1]
        
        # 정규화된 비율
        cg_ratio = (copper_price / gold_price) * 100
        
        if cg_ratio > 1.5:
            signal = '🟢 경기 확장'
            desc = f'구리/금 비율 {cg_ratio:.2f} - 리스크 온'
        elif cg_ratio < 0.8:
            signal = '🔴 경기 둔화'
            desc = f'구리/금 비율 {cg_ratio:.2f} - 리스크 오프'
        else:
            signal = '🟡 균형'
            desc = f'구리/금 비율 {cg_ratio:.2f} - 중립'
        
        ratios['copper_gold'] = {
            'ratio': cg_ratio,
            'signal': signal,
            'description': desc
        }
    
    return ratios

# ============================================================================
# 차트 렌더링
# ============================================================================

def render_dxy_analysis_chart(macro_data):
    """DXY 분석 차트"""
    if 'dxy' not in macro_data:
        return
    
    hist = macro_data['dxy']['history']
    
    fig = go.Figure()
    
    # 가격
    fig.add_trace(go.Scatter(
        x=hist.index,
        y=hist['Close'],
        mode='lines',
        name='DXY',
        line=dict(color='#2E86AB', width=2.5)
    ))
    
    # MA50
    ma50 = hist['Close'].rolling(window=50).mean()
    fig.add_trace(go.Scatter(
        x=hist.index,
        y=ma50,
        mode='lines',
        name='MA50',
        line=dict(color='orange', width=1.5, dash='dash')
    ))
    
    # MA200
    ma200 = hist['Close'].rolling(window=200).mean()
    fig.add_trace(go.Scatter(
        x=hist.index,
        y=ma200,
        mode='lines',
        name='MA200',
        line=dict(color='red', width=1.5, dash='dot')
    ))
    
    fig.update_layout(
        title='DXY (달러지수) 추세 분석',
        xaxis_title='날짜',
        yaxis_title='DXY',
        height=400,
        hovermode='x unified',
        legend=dict(x=0, y=1, bgcolor='rgba(255,255,255,0.8)')
    )
    
    return fig

def render_commodity_comparison(commodity_data):
    """원자재 상대 성과 비교"""
    fig = go.Figure()
    
    for key, data in commodity_data.items():
        hist = data['history']
        info = data['info']
        
        # 정규화
        normalized = (hist['Close'] / hist['Close'].iloc[0]) * 100
        
        fig.add_trace(go.Scatter(
            x=hist.index,
            y=normalized,
            mode='lines',
            name=info['name'],
            line=dict(color=info['color'], width=2)
        ))
    
    fig.update_layout(
        title='원자재 상대 성과 비교 (시작점 = 100)',
        xaxis_title='날짜',
        yaxis_title='상대 가격',
        height=500,
        hovermode='x unified'
    )
    
    return fig

def render_hrp_dendrogram(returns_df):
    """HRP 덴드로그램"""
    corr_matrix = returns_df.corr()
    dist_matrix = np.sqrt((1 - corr_matrix) / 2)
    linkage_matrix = linkage(squareform(dist_matrix.values), method='single')
    
    fig = go.Figure()
    
    # 덴드로그램 데이터 생성
    dend = dendrogram(linkage_matrix, labels=returns_df.columns.tolist(), no_plot=True)
    
    # Plotly로 렌더링
    for i, (x, y) in enumerate(zip(dend['icoord'], dend['dcoord'])):
        fig.add_trace(go.Scatter(
            x=x,
            y=y,
            mode='lines',
            line=dict(color='#667eea', width=2),
            showlegend=False,
            hoverinfo='skip'
        ))
    
    # 레이블 추가
    for i, label in enumerate(dend['ivl']):
        fig.add_annotation(
            x=10 * (i + 0.5),
            y=0,
            text=label,
            showarrow=False,
            yshift=-10,
            textangle=-45
        )
    
    fig.update_layout(
        title='HRP 계층적 클러스터링 (자산 간 상관관계)',
        xaxis=dict(showticklabels=False),
        yaxis_title='거리 (Distance)',
        height=400
    )
    
    return fig

# ============================================================================
# 메인 함수
# ============================================================================

def main():
    st.markdown('<h1 class="main-header">⚡ 고급 원자재 포트폴리오 최적화</h1>', unsafe_allow_html=True)
    st.markdown("""
    <div style='text-align: center; color: #666; margin-bottom: 2rem;'>
    DXY 필터 기반 전술적 자산 배분 (TAA) + 계층적 위험 균형 (HRP) 최적화
    </div>
    """, unsafe_allow_html=True)
    
    # 사이드바
    with st.sidebar:
        st.header("⚙️ 설정")
        
        lookback_days = st.select_slider(
            "데이터 기간",
            options=[90, 180, 365, 730],
            value=365
        )
        
        show_hrp = st.checkbox("HRP 최적화 표시", value=True)
        show_dendro = st.checkbox("클러스터링 덴드로그램", value=False)
        show_ratios = st.checkbox("비율 지표 분석", value=True)
        
        st.divider()
        
        # 포트폴리오 설정
        st.subheader("📊 포트폴리오 구성")
        
        selected_commodities = st.multiselect(
            "원자재 선택",
            options=list(COMMODITY_ETFS.keys()),
            default=['gold', 'silver', 'copper', 'oil_dbo', 'rare_earth', 'corn'],
            format_func=lambda x: COMMODITY_ETFS[x]['name']
        )
        
        st.divider()
        
        if st.button("🔄 새로고침"):
            st.cache_data.clear()
            st.rerun()
        
        st.caption(f"업데이트: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # 데이터 로드
    with st.spinner("📊 시장 데이터 로딩 중..."):
        commodity_data = fetch_commodity_data(lookback_days)
        macro_data = fetch_macro_data()
    
    if not commodity_data:
        st.error("데이터 로드 실패")
        return
    
    # 선택된 원자재만 필터링
    filtered_data = {k: v for k, v in commodity_data.items() if k in selected_commodities}
    
    if not filtered_data:
        st.warning("선택된 원자재가 없습니다.")
        return
    
    # === DXY 레짐 분석 ===
    st.subheader("🎯 DXY 거시 레짐 필터")
    
    dxy_regime = analyze_dxy_regime(macro_data)
    
    if dxy_regime:
        regime_class = {
            'strong_bullish': 'regime-bullish',
            'bullish': 'regime-bullish',
            'neutral': 'regime-neutral',
            'bearish': 'regime-bearish',
            'strong_bearish': 'regime-bearish'
        }.get(dxy_regime['regime_type'], 'regime-neutral')
        
        # 안전한 포맷팅을 위해 먼저 값 준비
        ma50_text = f"{dxy_regime['ma50']:.2f}" if dxy_regime['ma50'] is not None else 'N/A'
        ma200_text = f"{dxy_regime['ma200']:.2f}" if dxy_regime['ma200'] is not None else 'N/A'
        rsi_text = f"{dxy_regime['rsi']:.1f}" if dxy_regime['rsi'] is not None else 'N/A'
        
        st.markdown(f"""
        <div class="{regime_class}">
            <div style="font-size: 2rem; margin-bottom: 0.5rem;">{dxy_regime['regime_name']}</div>
            <div style="font-size: 1.1rem; margin-bottom: 1rem;">{dxy_regime['recommendation']}</div>
            <div style="font-size: 0.9rem;">DXY: {dxy_regime['current_price']:.2f} | MA50: {ma50_text} | MA200: {ma200_text} | RSI: {rsi_text}</div>
        </div>
        """, unsafe_allow_html=True)
        
        # 신호 상세
        with st.expander("📋 DXY 신호 상세", expanded=False):
            for signal in dxy_regime['signals']:
                st.write(f"• {signal}")
        
        # DXY 차트
        col1, col2 = st.columns([2, 1])
        
        with col1:
            dxy_chart = render_dxy_analysis_chart(macro_data)
            if dxy_chart:
                st.plotly_chart(dxy_chart, use_container_width=True)
        
        with col2:
            st.markdown("### 📊 거시경제 지표")
            
            if 'vix' in macro_data:
                vix = macro_data['vix']
                st.metric("VIX", f"{vix['current']:.1f}", f"{vix['change_pct']:+.2f}%")
            
            if 'us10y' in macro_data:
                us10y = macro_data['us10y']
                st.metric("미10년물", f"{us10y['current']:.2f}%", f"{us10y['change_pct']:+.2f}%")
            
            if 'spx' in macro_data:
                spx = macro_data['spx']
                st.metric("S&P500", f"{spx['current']:.2f}", f"{spx['change_pct']:+.2f}%")
        
        st.divider()
    
    # === 비율 지표 분석 ===
    if show_ratios:
        st.subheader("📐 핵심 비율 지표")
        
        ratios = calculate_ratios(filtered_data)
        
        if ratios:
            cols = st.columns(len(ratios))
            
            for idx, (ratio_name, ratio_data) in enumerate(ratios.items()):
                with cols[idx]:
                    st.markdown(f"""
                    <div class="metric-card">
                        <h4 style="margin: 0 0 0.5rem 0;">{ratio_name.replace('_', '/').upper()}</h4>
                        <div style="font-size: 2rem; font-weight: bold; color: #667eea;">{ratio_data['ratio']:.2f}</div>
                        <div style="font-size: 1.1rem; margin: 0.5rem 0;">{ratio_data['signal']}</div>
                        <div style="font-size: 0.85rem; color: #666;">{ratio_data['description']}</div>
                    </div>
                    """, unsafe_allow_html=True)
        else:
            st.info("비율 계산을 위한 데이터가 부족합니다.")
        
        st.divider()
    
    # === HRP 최적화 ===
    if show_hrp and len(filtered_data) >= 3:
        st.subheader("🧮 HRP (계층적 위험 균형) 포트폴리오 최적화")
        
        try:
            # 수익률 데이터 준비
            returns_data = {}
            min_length = min([len(data['history']) for data in filtered_data.values()])
            
            for key, data in filtered_data.items():
                hist = data['history']['Close'].iloc[-min_length:]
                returns = hist.pct_change().dropna()
                returns_data[data['info']['name']] = returns
            
            returns_df = pd.DataFrame(returns_data)
            returns_df = returns_df.dropna()
            
            if len(returns_df) < 30:
                st.warning("데이터가 부족하여 HRP 최적화를 수행할 수 없습니다. (최소 30일 필요)")
            else:
                # HRP 가중치 계산
                hrp_weights = calculate_hrp_weights(returns_df)
                
                # 결과 시각화
                col1, col2 = st.columns([1, 1])
                
                with col1:
                    # 가중치 파이 차트
                    colors = []
                    for name in hrp_weights.index:
                        # 이름으로 원래 키 찾기
                        for k, v in filtered_data.items():
                            if v['info']['name'] == name:
                                colors.append(v['info']['color'])
                                break
                    
                    fig_pie = go.Figure(data=[go.Pie(
                        labels=hrp_weights.index,
                        values=hrp_weights.values,
                        hole=0.4,
                        marker=dict(colors=colors) if colors else None
                    )])
                    
                    fig_pie.update_layout(
                        title='HRP 최적 가중치 배분',
                        height=400
                    )
                    
                    st.plotly_chart(fig_pie, use_container_width=True)
                
                with col2:
                    # 가중치 테이블
                    st.markdown("### 📊 최적 가중치")
                    
                    weight_df = pd.DataFrame({
                        '자산': hrp_weights.index,
                        '가중치': [f"{w*100:.2f}%" for w in hrp_weights.values],
                        '추천 금액 ($10K)': [f"${w*10000:.0f}" for w in hrp_weights.values]
                    })
                    
                    st.dataframe(weight_df, use_container_width=True, hide_index=True)
                    
                    # 포트폴리오 통계
                    st.markdown("### 📈 포트폴리오 통계")
                    
                    portfolio_returns = (returns_df * hrp_weights).sum(axis=1)
                    annual_return = portfolio_returns.mean() * 252
                    annual_vol = portfolio_returns.std() * np.sqrt(252)
                    sharpe = annual_return / annual_vol if annual_vol > 0 else 0
                    
                    stat_col1, stat_col2, stat_col3 = st.columns(3)
                    with stat_col1:
                        st.metric("연간 수익률", f"{annual_return*100:.2f}%")
                    with stat_col2:
                        st.metric("연간 변동성", f"{annual_vol*100:.2f}%")
                    with stat_col3:
                        st.metric("샤프 비율", f"{sharpe:.2f}")
                
                # 덴드로그램
                if show_dendro:
                    st.markdown("### 🌳 계층적 클러스터링")
                    dendro_fig = render_hrp_dendrogram(returns_df)
                    st.plotly_chart(dendro_fig, use_container_width=True)
                
                # 상관관계 히트맵
                st.markdown("### 🔥 자산 간 상관관계")
                
                corr_matrix = returns_df.corr()
                
                fig_corr = go.Figure(data=go.Heatmap(
                    z=corr_matrix.values,
                    x=corr_matrix.columns,
                    y=corr_matrix.columns,
                    colorscale='RdBu',
                    zmid=0,
                    text=np.round(corr_matrix.values, 2),
                    texttemplate='%{text}',
                    textfont={"size": 10}
                ))
                
                fig_corr.update_layout(
                    title='자산 간 상관관계 행렬',
                    height=500
                )
                
                st.plotly_chart(fig_corr, use_container_width=True)
                
        except Exception as e:
            st.error(f"HRP 최적화 실패: {str(e)}")
            import traceback
            st.code(traceback.format_exc())
        
        st.divider()
    
    # === 원자재 상대 성과 ===
    st.subheader("📈 원자재 상대 성과 분석")
    
    compare_fig = render_commodity_comparison(filtered_data)
    st.plotly_chart(compare_fig, use_container_width=True)
    
    # 성과 테이블
    performance_data = []
    
    for key, data in filtered_data.items():
        hist = data['history']['Close']
        current = hist.iloc[-1]
        start = hist.iloc[0]
        
        month_ago = hist.iloc[-20] if len(hist) >= 20 else start
        quarter_ago = hist.iloc[-60] if len(hist) >= 60 else start
        
        performance_data.append({
            '원자재': data['info']['name'],
            '카테고리': data['info']['category'],
            '현재가': f"${current:.2f}",
            '1개월': f"{((current-month_ago)/month_ago*100):+.2f}%",
            '3개월': f"{((current-quarter_ago)/quarter_ago*100):+.2f}%",
            'YTD': f"{((current-start)/start*100):+.2f}%"
        })
    
    perf_df = pd.DataFrame(performance_data)
    st.dataframe(perf_df, use_container_width=True, hide_index=True)
    
    st.divider()
    
    # === 전략 요약 ===
    st.subheader("💼 통합 전략 요약")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        <div class="info-box">
            <h4>✅ DXY 기반 TAA 전략</h4>
            <p><strong>목적:</strong> 거시경제 레짐에 따른 리스크 관리</p>
            <ul>
                <li><strong>달러 강세 시:</strong> 원자재 비중 축소, 현금/국채 확대</li>
                <li><strong>달러 약세 시:</strong> 원자재 비중 확대, 공격적 배분</li>
                <li><strong>중립 시:</strong> 현 비중 유지, HRP 최적화 활용</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="info-box">
            <h4>🎯 HRP 기반 포트폴리오 구축</h4>
            <p><strong>목적:</strong> 상관관계 기반 리스크 분산</p>
            <ul>
                <li>낮은 상관관계 자산에 높은 가중치</li>
                <li>클러스터 내 리스크 균형 유지</li>
                <li>구조적 위험 최소화 (콘탱고, 지정학)</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    # 경고 사항
    st.markdown("""
    <div class="warning-box">
        <h4>⚠️ 주요 리스크 및 유의사항</h4>
        <ul>
            <li><strong>에너지 (원유):</strong> 콘탱고 리스크 - DBO와 같은 최적화 ETF 선호</li>
            <li><strong>희귀원소 (REMX):</strong> 극심한 변동성 (37% 추적오차) - 소규모 전략적 배분만 권장</li>
            <li><strong>농산물:</strong> 계절성 고려 필수 - 수확기 진입 회피</li>
            <li><strong>LP 호가 시간:</strong> 한국 시간 09:05 이후 거래 권장</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)
    
    # 면책 조항
    st.divider()
    st.caption("⚠️ 본 대시보드는 정보 제공 목적이며, 투자 권유가 아닙니다. 모든 투자 결정의 책임은 투자자 본인에게 있습니다.")

if __name__ == "__main__":
    main()