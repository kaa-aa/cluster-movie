import numpy as np
import pandas as pd
import plotly.express as px
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import streamlit as st

# 1. 페이지 기본 설정 (아이콘 및 탭 제목)
st.set_page_config(
    page_title="영화 유형 나누기",
    page_icon="🎬",
    layout="wide"
)

st.title("🎬 영화 유형 나누기 (K-Means 군집 분석)")

# 2. 데이터 로드 및 전처리 함수
@st.cache_data
def load_and_preprocess_data():
    url = "https://raw.githubusercontent.com/it-teacher-go/data-share-/main/kobis_movies.csv"
    df = pd.read_csv(url, encoding="utf-8")
    
    total_raw_count = len(df)
    
    # 필수 속성 결측치 제거 및 first_week_audi가 0인 데이터 제거
    req_cols = ['first_scrn', 'total_audi', 'days_in_top10', 'first_week_audi', 'movieNm']
    df = df.dropna(subset=req_cols).copy()
    df = df[df['first_week_audi'] > 0].copy()
    
    # 파생 변수 및 변환 속성 생성
    # 0 이하 값이 들어가지 않도록 clip 처리 후 상용로그(log10) 적용
    df['log_first_scrn'] = np.log10(df['first_scrn'].clip(lower=1))
    df['log_total_audi'] = np.log10(df['total_audi'].clip(lower=1))
    df['days_in_top10'] = df['days_in_top10']
    
    # 롱런 지수: 누적 관객 / 첫 주 관객 (20 초과는 20으로 클리핑)
    df['long_run_idx'] = (df['total_audi'] / df['first_week_audi']).clip(upper=20)
    
    used_count = len(df)
    return df, total_raw_count, used_count

df, total_raw_count, used_count = load_and_preprocess_data()

# 편수 안내 한 줄 표시
st.write(f"📊 **전체 데이터 수:** {total_raw_count:,}편 | **분석에 사용된 편수:** {used_count:,}편")
st.markdown("---")

# 3. 군집화에 사용할 속성 선택 (화면 상단)
feature_options = {
    "스크린 수(로그)": "log_first_scrn",
    "누적 관객 수(로그)": "log_total_audi",
    "10위권 일수": "days_in_top10",
    "롱런 지수": "long_run_idx"
}

selected_feature_names = st.multiselect(
    "군집화에 사용할 속성을 선택하세요 (2개 이상 선택):",
    options=list(feature_options.keys()),
    default=list(feature_options.keys())
)

if len(selected_feature_names) < 2:
    st.error("속성을 최소 2개 이상 선택해야 군집화를 진행할 수 있습니다.")
    st.stop()

selected_features = [feature_options[name] for name in selected_feature_names]

# 4. 표준화 및 K-Means 군집화 (k=3, 난수 고정)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(df[selected_features])

kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
raw_clusters = kmeans.fit_predict(X_scaled)
df['temp_cluster'] = raw_clusters

# 누적 관객 평균이 큰 묶음부터 ㉮, ㉯, ㉰ 부여
cluster_audi_mean = df.groupby('temp_cluster')['total_audi'].mean().sort_values(ascending=False)
label_map = {old_label: new_label for old_label, new_label in zip(cluster_audi_mean.index, ['㉮', '㉯', '㉰'])}
df['cluster'] = df['temp_cluster'].map(label_map)

# 카테고리 순서 정렬
cluster_order = ['㉮', '㉯', '㉰']

# 5. 2차원 산점도 시각화
st.subheader("📌 2차원 산점도")
col1, col2 = st.columns(2)

with col1:
    x_axis_2d_name = st.selectbox("가로축(X) 속성 선택:", options=selected_feature_names, index=0)
with col2:
    y_axis_2d_name = st.selectbox("세로축(Y) 속성 선택:", options=selected_feature_names, index=min(1, len(selected_feature_names)-1))

fig_2d = px.scatter(
    df,
    x=feature_options[x_axis_2d_name],
    y=feature_options[y_axis_2d_name],
    color='cluster',
    category_orders={'cluster': cluster_order},
    hover_name='movieNm',
    labels={
        feature_options[x_axis_2d_name]: x_axis_2d_name,
        feature_options[y_axis_2d_name]: y_axis_2d_name,
        'cluster': '묶음'
    },
    title=f"2D 산점도: {x_axis_2d_name} vs {y_axis_2d_name}"
)
st.plotly_chart(fig_2d, use_container_width=True)

# 6. 3차원 산점도 시각화
st.subheader("📌 3차원 산점도")
if len(selected_feature_names) >= 3:
    col3, col4, col5 = st.columns(3)
    with col3:
        x_axis_3d_name = st.selectbox("3D X축 속성:", options=selected_feature_names, index=0, key="3d_x")
    with col4:
        y_axis_3d_name = st.selectbox("3D Y축 속성:", options=selected_feature_names, index=1, key="3d_y")
    with col5:
        z_axis_3d_name = st.selectbox("3D Z축 속성:", options=selected_feature_names, index=2, key="3d_z")

    fig_3d = px.scatter_3d(
        df,
        x=feature_options[x_axis_3d_name],
        y=feature_options[y_axis_3d_name],
        z=feature_options[z_axis_3d_name],
        color='cluster',
        category_orders={'cluster': cluster_order},
        hover_name='movieNm',
        labels={
            feature_options[x_axis_3d_name]: x_axis_3d_name,
            feature_options[y_axis_3d_name]: y_axis_3d_name,
            feature_options[z_axis_3d_name]: z_axis_3d_name,
            'cluster': '묶음'
        },
        title=f"3D 산점도: {x_axis_3d_name} vs {y_axis_3d_name} vs {z_axis_3d_name}"
    )
    fig_3d.update_traces(marker=dict(size=3))  # 점 크기 작게 설정
    st.plotly_chart(fig_3d, use_container_width=True)
else:
    st.info("💡 3차원 산점도를 출력하려면 위에서 3개 이상의 속성을 선택해 주세요.")

st.markdown("---")

# 7. 묶음별 요약 통계표 (원래 단위 평균)
st.subheader("📊 묶음별 영화 편수 및 속성 평균 (원래 단위)")

summary_df = df.groupby('cluster').agg(
    편수=('movieNm', 'count'),
    스크린_수_평균=('first_scrn', 'mean'),
    누적_관객_평균=('total_audi', 'mean'),
    TOP10_일수_평균=('days_in_top10', 'mean'),
    롱런_지수_평균=('long_run_idx', 'mean')
).reindex(cluster_order)

# 소수점 및 숫자 포맷팅
summary_df_display = summary_df.copy()
summary_df_display['편수'] = summary_df_display['편수'].map('{:,}편'.format)
summary_df_display['스크린_수_평균'] = summary_df_display['스크린_수_평균'].map('{:,.1f}개'.format)
summary_df_display['누적_관객_평균'] = summary_df_display['누적_관객_평균'].map('{:,.0f}명'.format)
summary_df_display['TOP10_일수_평균'] = summary_df_display['TOP10_일수_평균'].map('{:,.1f}일'.format)
summary_df_display['롱런_지수_평균'] = summary_df_display['롱런_지수_평균'].map('{:,.2f}'.format)

st.dataframe(summary_df_display, use_container_width=True)

# 8. 묶음별 누적 관객수 상위 5개 영화 목록
st.subheader("🏆 묶음별 누적 관객 수 TOP 5 영화")

cols = st.columns(3)
for idx, cluster_label in enumerate(cluster_order):
    with cols[idx]:
        st.markdown(f"### 묶음 {cluster_label}")
        top5_movies = (
            df[df['cluster'] == cluster_label]
            .sort_values(by='total_audi', ascending=False)
            .head(5)
        )
        for rank, (_, row) in enumerate(top5_movies.iterrows(), start=1):
            st.write(f"**{rank}. {row['movieNm']}** ({row['total_audi']:,}명)")
