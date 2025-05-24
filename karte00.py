import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import re
import io
import plotly.express as px
import plotly.graph_objects as go
from collections import Counter, defaultdict
import networkx as nx
import japanize_matplotlib
import spacy
from datetime import datetime, timedelta
import json

# spaCyの日本語モデルを読み込み
@st.cache_resource
def load_spacy_model():
    try:
        nlp = spacy.load("ja_core_news_sm")
        return nlp
    except OSError:
        st.error("spaCyの日本語モデル 'ja_core_news_sm' がインストールされていません。")
        st.info("以下のコマンドでインストールしてください: python -m spacy download ja_core_news_sm")
        return None

# アプリのタイトルとスタイル設定
st.set_page_config(page_title="授業記録・カルテ分析ツール", layout="wide")
st.title("📚 授業記録・カルテ分析ツール（教師支援版）")

# spaCyモデルの読み込み
nlp = load_spacy_model()

if nlp is None:
    st.stop()

# サイドバーの設定
st.sidebar.header("📁 データアップロード")

# 授業記録（CSV）のアップロード
uploaded_csv = st.sidebar.file_uploader("授業記録CSVファイルをアップロード", type=["csv"])
# カルテ（テキスト）のアップロード
uploaded_txt = st.sidebar.file_uploader("カルテテキストファイルをアップロード", type=["txt"])

# カルテの形式を選択
karte_format = st.sidebar.radio(
    "カルテの形式を選択",
    ["#名前 メモ", "名前: メモ", "その他"]
)

# カルテのパターン設定
if karte_format == "#名前 メモ":
    karte_pattern = r"#([^\s]+)\s+(.*)"
elif karte_format == "名前: メモ":
    karte_pattern = r"([^:]+):\s*(.*)"
else:
    karte_pattern = st.sidebar.text_input("カルテのパターンを正規表現で入力", r"#([^\s]+)\s+(.*)")

# 分析タブの設定
tabs = st.tabs(["🎯 児童個別分析", "🤝 児童間関係分析", "📊 授業全体分析", "💡 教師への提案"])

# spaCyを使った文書処理関数
def process_text_with_spacy(text, min_length=2):
    """spaCyを使ってテキストを処理し、有用な単語を抽出"""
    doc = nlp(text)
    
    # ストップワードと除外する品詞
    stop_pos = ["ADP", "AUX", "CCONJ", "DET", "INTJ", "PART", "PRON", "PUNCT", "SCONJ", "SYM", "X"]
    
    words = []
    for token in doc:
        if (not token.is_stop and 
            not token.is_punct and 
            not token.is_space and
            len(token.text) >= min_length and
            token.pos_ not in stop_pos):
            words.append(token.lemma_)
    
    return words, doc

def analyze_student_interactions(df_class):
    """授業記録から児童間の相互作用を分析"""
    interactions = defaultdict(list)
    
    for i in range(len(df_class) - 1):
        current_speaker = df_class.iloc[i]['発言者']
        next_speaker = df_class.iloc[i + 1]['発言者']
        current_content = df_class.iloc[i]['発言内容']
        next_content = df_class.iloc[i + 1]['発言内容']
        
        # 教師以外の発言の連続を分析
        if current_speaker != '教師' and next_speaker != '教師':
            # 質問応答パターンの検出
            if any(word in current_content for word in ['？', '?', 'どう', 'なぜ', 'どこ', 'いつ', 'だれ']):
                interactions[current_speaker].append({
                    'type': '質問',
                    'target': next_speaker,
                    'content': current_content,
                    'response': next_content
                })
            
            # 同意・反対パターンの検出
            if any(word in next_content for word in ['そう', 'はい', '同じ', '賛成']):
                interactions[next_speaker].append({
                    'type': '同意',
                    'target': current_speaker,
                    'content': next_content
                })
            elif any(word in next_content for word in ['違う', 'でも', 'しかし', '反対']):
                interactions[next_speaker].append({
                    'type': '反対',
                    'target': current_speaker,
                    'content': next_content
                })
    
    return interactions

def generate_teaching_suggestions(df_class, df_karte, student_name):
    """個別の児童に対する教師への提案を生成"""
    suggestions = []
    
    # 発言データの分析
    student_statements = df_class[df_class['発言者'] == student_name]
    statement_count = len(student_statements)
    
    # カルテデータの分析
    student_karte = df_karte[df_karte['生徒名'] == student_name]
    
    # 発言頻度に基づく提案
    total_statements = len(df_class[df_class['発言者'] != '教師'])
    if total_statements > 0:
        participation_rate = statement_count / total_statements
        
        if participation_rate < 0.1:
            suggestions.append({
                'category': '参加促進',
                'suggestion': f'{student_name}さんの発言機会を意識的に作りましょう。指名や小グループでの活動を増やすことを検討してください。',
                'priority': 'high'
            })
        elif participation_rate > 0.3:
            suggestions.append({
                'category': '発言調整',
                'suggestion': f'{student_name}さんは積極的です。他の児童にも発言機会を回すよう配慮し、{student_name}さんには司会や発表役を任せることを検討してください。',
                'priority': 'medium'
            })
    
    # カルテ内容に基づく提案
    if len(student_karte) > 0:
        karte_text = ' '.join(student_karte['メモ'])
        
        # 注意が必要なキーワード
        attention_keywords = ['集中', '注意', '騒ぐ', '立ち歩く', '忘れ物']
        support_keywords = ['理解', '困る', '分からない', '質問']
        social_keywords = ['友達', '仲良し', '対立', '孤立']
        
        for keyword in attention_keywords:
            if keyword in karte_text:
                suggestions.append({
                    'category': '行動支援',
                    'suggestion': f'{student_name}さんの{keyword}に関する記録があります。座席配置や役割分担を工夫し、集中しやすい環境を整えましょう。',
                    'priority': 'high'
                })
        
        for keyword in support_keywords:
            if keyword in karte_text:
                suggestions.append({
                    'category': '学習支援',
                    'suggestion': f'{student_name}さんの学習面でのサポートが必要です。個別指導の時間を設けたり、理解度を確認する機会を増やしましょう。',
                    'priority': 'high'
                })
        
        for keyword in social_keywords:
            if keyword in karte_text:
                suggestions.append({
                    'category': '人間関係',
                    'suggestion': f'{student_name}さんの人間関係に注意が必要です。グループ活動の組み合わせを工夫し、良好な関係構築を支援しましょう。',
                    'priority': 'medium'
                })
    
    return suggestions

# データが読み込まれた場合の処理
if uploaded_csv is not None and uploaded_txt is not None:
    # CSVデータの読み込み
    try:
        df_class = pd.read_csv(uploaded_csv, encoding='utf-8')
    except UnicodeDecodeError:
        df_class = pd.read_csv(uploaded_csv, encoding='shift-jis')
    
    # カラム名の確認と修正
    if len(df_class.columns) >= 3:
        df_class.columns = ['発言番号', '発言者', '発言内容'] + list(df_class.columns[3:])
    else:
        st.error("CSVファイルには少なくとも3つのカラム（発言番号、発言者、発言内容）が必要です。")
        st.stop()
    
    # カルテデータの読み込み
    try:
        karte_text = uploaded_txt.read().decode('utf-8')
    except UnicodeDecodeError:
        karte_text = uploaded_txt.read().decode('shift-jis')
    
    # カルテデータの解析
    karte_entries = []
    for line in karte_text.split('\n'):
        if line.strip():
            match = re.match(karte_pattern, line.strip())
            if match:
                student_name = match.group(1)
                note = match.group(2)
                karte_entries.append({"生徒名": student_name, "メモ": note})
    
    df_karte = pd.DataFrame(karte_entries)
    
    # 授業記録から児童名を抽出
    students_in_class = df_class[df_class['発言者'] != '教師']['発言者'].unique()
    students_in_karte = df_karte['生徒名'].unique()
    all_students = sorted(set(list(students_in_class) + list(students_in_karte)))
    
    # タブ1: 児童個別分析
    with tabs[0]:
        st.header("🎯 児童個別分析")
        
        # 児童選択
        selected_student = st.selectbox("分析する児童を選択してください", all_students)
        
        if selected_student:
            col1, col2 = st.columns([1, 1])
            
            with col1:
                st.subheader(f"📝 {selected_student}さんのカルテ履歴")
                
                student_karte = df_karte[df_karte['生徒名'] == selected_student]
                
                if len(student_karte) > 0:
                    # カルテをページめくり形式で表示
                    if 'karte_page' not in st.session_state:
                        st.session_state.karte_page = 0
                    
                    total_pages = len(student_karte)
                    current_page = st.session_state.karte_page
                    
                    # ページナビゲーション
                    col_prev, col_info, col_next = st.columns([1, 2, 1])
                    
                    with col_prev:
                        if st.button("⬅️ 前", disabled=(current_page == 0)):
                            st.session_state.karte_page = max(0, current_page - 1)
                            st.rerun()
                    
                    with col_info:
                        st.write(f"📄 {current_page + 1} / {total_pages}")
                    
                    with col_next:
                        if st.button("➡️ 次", disabled=(current_page == total_pages - 1)):
                            st.session_state.karte_page = min(total_pages - 1, current_page + 1)
                            st.rerun()
                    
                    # 現在のページのカルテを表示
                    current_karte = student_karte.iloc[current_page]
                    
                    st.markdown(f"""
                    <div style="
                        border: 2px solid #ddd; 
                        border-radius: 10px; 
                        padding: 20px; 
                        margin: 10px 0; 
                        background-color: #f9f9f9;
                        min-height: 150px;
                    ">
                        <h4 style="color: #333; margin-bottom: 15px;">📋 記録 {current_page + 1}</h4>
                        <p style="font-size: 16px; line-height: 1.6; color: #555;">
                            {current_karte['メモ']}
                        </p>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # 全カルテ一覧表示オプション
                    if st.checkbox("📚 全カルテを一覧表示"):
                        st.dataframe(student_karte, use_container_width=True)
                else:
                    st.info(f"{selected_student}さんのカルテ記録はありません。")
            
            with col2:
                st.subheader(f"💬 {selected_student}さんの授業での発言")
                
                student_statements = df_class[df_class['発言者'] == selected_student]
                
                if len(student_statements) > 0:
                    # 発言の統計情報
                    total_statements = len(df_class[df_class['発言者'] != '教師'])
                    participation_rate = len(student_statements) / total_statements if total_statements > 0 else 0
                    
                    col_stat1, col_stat2 = st.columns(2)
                    with col_stat1:
                        st.metric("発言回数", len(student_statements))
                    with col_stat2:
                        st.metric("参加率", f"{participation_rate:.1%}")
                    
                    # 発言内容の表示
                    st.dataframe(student_statements[['発言番号', '発言内容']], use_container_width=True)
                    
                    # 発言内容のキーワード分析
                    st.subheader("🔍 発言キーワード分析")
                    all_speech = ' '.join(student_statements['発言内容'])
                    words, _ = process_text_with_spacy(all_speech)
                    
                    if words:
                        word_counts = Counter(words).most_common(10)
                        keywords = [word for word, count in word_counts]
                        st.write("**よく使う言葉:** " + "、".join(keywords))
                else:
                    st.info(f"{selected_student}さんの発言記録はありません。")
            
            # 児童間関係の記録・編集機能
            st.subheader(f"🤝 {selected_student}さんの人間関係")
            
            # セッション状態で関係データを管理
            if 'relationships' not in st.session_state:
                st.session_state.relationships = {}
            
            # 関係タイプの選択
            relationship_types = ["仲良し", "対立", "意見が同じ", "教え合い", "競争関係", "その他"]
            
            col_rel1, col_rel2, col_rel3 = st.columns([2, 2, 1])
            
            with col_rel1:
                target_student = st.selectbox(
                    "関係を記録する相手を選択",
                    [s for s in all_students if s != selected_student],
                    key=f"target_{selected_student}"
                )
            
            with col_rel2:
                relationship_type = st.selectbox(
                    "関係の種類",
                    relationship_types,
                    key=f"rel_type_{selected_student}"
                )
            
            with col_rel3:
                if st.button("記録", key=f"add_rel_{selected_student}"):
                    if selected_student not in st.session_state.relationships:
                        st.session_state.relationships[selected_student] = []
                    
                    st.session_state.relationships[selected_student].append({
                        'target': target_student,
                        'type': relationship_type,
                        'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M")
                    })
                    st.success(f"{selected_student}さんと{target_student}さんの関係「{relationship_type}」を記録しました。")
            
            # 記録された関係の表示
            if selected_student in st.session_state.relationships:
                st.write("**記録された関係:**")
                for i, rel in enumerate(st.session_state.relationships[selected_student]):
                    col_show, col_del = st.columns([4, 1])
                    with col_show:
                        st.write(f"• {rel['target']} - {rel['type']} ({rel['timestamp']})")
                    with col_del:
                        if st.button("削除", key=f"del_{selected_student}_{i}"):
                            st.session_state.relationships[selected_student].pop(i)
                            st.rerun()
    
    # タブ2: 児童間関係分析
    with tabs[1]:
        st.header("🤝 児童間関係分析")
        
        # 授業記録から自動検出された相互作用
        st.subheader("📊 授業中の相互作用パターン")
        
        interactions = analyze_student_interactions(df_class)
        
        if interactions:
            # 相互作用のネットワーク図
            G = nx.Graph()
            
            # ノードの追加
            for student in students_in_class:
                G.add_node(student)
            
            # エッジの追加（相互作用に基づく）
            edge_data = []
            for student, student_interactions in interactions.items():
                for interaction in student_interactions:
                    target = interaction['target']
                    interaction_type = interaction['type']
                    
                    if G.has_edge(student, target):
                        G[student][target]['weight'] += 1
                        G[student][target]['types'].append(interaction_type)
                    else:
                        G.add_edge(student, target, weight=1, types=[interaction_type])
                    
                    edge_data.append({
                        'from': student,
                        'to': target,
                        'type': interaction_type,
                        'content': interaction['content']
                    })
            
            # ネットワーク図の描画
            if len(G.nodes()) > 0:
                pos = nx.spring_layout(G, seed=42)
                
                fig, ax = plt.subplots(figsize=(12, 8))
                
                # ノードの描画
                nx.draw_networkx_nodes(G, pos, node_size=1000, node_color='lightblue', ax=ax)
                
                # エッジの描画
                if len(G.edges()) > 0:
                    edge_weights = [G[u][v]['weight'] for u, v in G.edges()]
                    max_weight = max(edge_weights) if edge_weights else 1
                    edge_widths = [2 + 3 * (weight / max_weight) for weight in edge_weights]
                    
                    nx.draw_networkx_edges(G, pos, width=edge_widths, alpha=0.7, edge_color='gray', ax=ax)
                
                # ラベルの描画
                nx.draw_networkx_labels(G, pos, font_size=10, font_family='sans-serif', ax=ax)
                
                plt.axis('off')
                plt.title("児童間の相互作用ネットワーク")
                st.pyplot(fig)
            
            # 相互作用の詳細表示
            st.subheader("📝 相互作用の詳細")
            
            if edge_data:
                df_interactions = pd.DataFrame(edge_data)
                
                # 相互作用タイプ別の集計
                interaction_summary = df_interactions.groupby(['from', 'to', 'type']).size().reset_index(name='count')
                st.dataframe(interaction_summary, use_container_width=True)
        else:
            st.info("授業記録から明確な児童間相互作用を検出できませんでした。")
        
        # 手動で記録された関係の可視化
        st.subheader("👥 記録された人間関係")
        
        if 'relationships' in st.session_state and st.session_state.relationships:
            # 関係データをネットワーク図で表示
            G_manual = nx.Graph()
            
            # 関係データからエッジを作成
            for student, relationships in st.session_state.relationships.items():
                for rel in relationships:
                    G_manual.add_edge(student, rel['target'], type=rel['type'])
            
            if len(G_manual.nodes()) > 0:
                pos = nx.spring_layout(G_manual, seed=42)
                
                fig, ax = plt.subplots(figsize=(10, 8))
                
                # 関係タイプ別の色分け
                edge_colors = []
                for u, v in G_manual.edges():
                    rel_type = G_manual[u][v]['type']
                    if rel_type == '仲良し':
                        edge_colors.append('green')
                    elif rel_type == '対立':
                        edge_colors.append('red')
                    elif rel_type == '意見が同じ':
                        edge_colors.append('blue')
                    elif rel_type == '教え合い':
                        edge_colors.append('orange')
                    else:
                        edge_colors.append('gray')
                
                nx.draw_networkx_nodes(G_manual, pos, node_size=800, node_color='lightcoral', ax=ax)
                nx.draw_networkx_edges(G_manual, pos, edge_color=edge_colors, width=2, ax=ax)
                nx.draw_networkx_labels(G_manual, pos, font_size=9, ax=ax)
                
                plt.axis('off')
                plt.title("記録された人間関係")
                st.pyplot(fig)
                
                # 凡例
                st.markdown("""
                **関係の色分け:**
                - 🟢 緑: 仲良し
                - 🔴 赤: 対立
                - 🔵 青: 意見が同じ
                - 🟠 オレンジ: 教え合い
                - ⚫ グレー: その他
                """)
        else:
            st.info("まだ人間関係が記録されていません。「児童個別分析」タブで関係を記録してください。")
    
    # タブ3: 授業全体分析
    with tabs[2]:
        st.header("📊 授業全体分析")
        
        # 発言パターンの分析
        st.subheader("💬 発言パターン分析")
        
        # 発言者の分布
        speaker_counts = df_class['発言者'].value_counts()
        
        fig = px.bar(
            x=speaker_counts.index,
            y=speaker_counts.values,
            labels={'x': '発言者', 'y': '発言回数'},
            title="発言者別の発言回数"
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # 発言の時系列パターン
        st.subheader("⏰ 発言の時系列パターン")
        
        # 教師と児童の発言パターン
        df_class['発言者タイプ'] = df_class['発言者'].apply(lambda x: '教師' if x == '教師' else '児童')
        
        fig = px.scatter(
            df_class,
            x='発言番号',
            y='発言者',
            color='発言者タイプ',
            title="授業中の発言パターン",
            hover_data=['発言内容']
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # 授業の流れ分析
        st.subheader("📈 授業の流れ分析")
        
        # 連続する教師発言の検出（説明が長い部分）
        teacher_sequences = []
        current_sequence = 0
        
        for i, row in df_class.iterrows():
            if row['発言者'] == '教師':
                current_sequence += 1
            else:
                if current_sequence > 0:
                    teacher_sequences.append(current_sequence)
                current_sequence = 0
        
        if current_sequence > 0:
            teacher_sequences.append(current_sequence)
        
        if teacher_sequences:
            avg_teacher_sequence = np.mean(teacher_sequences)
            max_teacher_sequence = max(teacher_sequences)
            
            col1, col2 = st.columns(2)
            with col1:
                st.metric("平均連続教師発言数", f"{avg_teacher_sequence:.1f}")
            with col2:
                st.metric("最大連続教師発言数", f"{max_teacher_sequence}")
            
            if max_teacher_sequence > 5:
                st.warning("⚠️ 教師の連続発言が長い部分があります。児童の参加機会を増やすことを検討してください。")
        
        # 児童の参加度分析
        st.subheader("👥 児童参加度分析")
        
        student_participation = df_class[df_class['発言者'] != '教師']['発言者'].value_counts()
        
        if len(student_participation) > 0:
            # 参加度の分布
            fig = px.histogram(
                x=student_participation.values,
                nbins=10,
                title="児童の発言回数分布",
                labels={'x': '発言回数', 'y': '児童数'}
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # 参加度の統計
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("発言した児童数", len(student_participation))
            with col2:
                st.metric("平均発言回数", f"{student_participation.mean():.1f}")
            with col3:
                silent_students = len([s for s in students_in_class if s not in student_participation.index])
                st.metric("発言なし児童数", silent_students)
    
    # タブ4: 教師への提案
    with tabs[3]:
        st.header("💡 教師への提案")
        
        # 全体的な授業改善提案
        st.subheader("🎯 授業全体への提案")
        
        suggestions = []
        
        # 参加度に基づく提案
        student_participation = df_class[df_class['発言者'] != '教師']['発言者'].value_counts()
        silent_students = [s for s in students_in_class if s not in student_participation.index]
        
        if len(silent_students) > 0:
            suggestions.append({
                'category': '参加促進',
                'suggestion': f"発言していない児童が{len(silent_students)}名います（{', '.join(silent_students)}）。小グループ活動や指名を活用して参加を促しましょう。",
                'priority': 'high'
            })
        
        # 発言の偏りに基づく提案
        if len(student_participation) > 0:
            participation_std = student_participation.std()
            if participation_std > 3:
                suggestions.append({
                    'category': '発言調整',
                    'suggestion': "発言回数に大きな偏りがあります。発言機会を均等に配分する工夫を検討してください。",
                    'priority': 'medium'
                })
        
        # 教師の連続発言に基づく提案
        teacher_sequences = []
        current_sequence = 0
        
        for i, row in df_class.iterrows():
            if row['発言者'] == '教師':
                current_sequence += 1
            else:
                if current_sequence > 0:
                    teacher_sequences.append(current_sequence)
                current_sequence = 0
        
        if teacher_sequences and max(teacher_sequences) > 5:
            suggestions.append({
                'category': '授業構成',
                'suggestion': "教師の連続発言が長い部分があります。児童との対話を増やし、理解度を確認しながら進めることを推奨します。",
                'priority': 'medium'
            })
        
        # 提案の表示
        for suggestion in suggestions:
            priority_color = {
                'high': '🔴',
                'medium': '🟡',
                'low': '🟢'
            }
            
            st.markdown(f"""
            <div style="
                border-left: 4px solid {'#ff4444' if suggestion['priority'] == 'high' else '#ffaa00' if suggestion['priority'] == 'medium' else '#44ff44'};
                padding: 15px;
                margin: 10px 0;
                background-color: #f8f9fa;
                border-radius: 5px;
            ">
                <h4>{priority_color[suggestion['priority']]} {suggestion['category']}</h4>
                <p>{suggestion['suggestion']}</p>
            </div>
            """, unsafe_allow_html=True)
        
        # 個別児童への提案
        st.subheader("👤 個別児童への提案")
        
        selected_student_for_suggestion = st.selectbox(
            "提案を見たい児童を選択",
            all_students,
            key="suggestion_student"
        )
        
        if selected_student_for_suggestion:
            individual_suggestions = generate_teaching_suggestions(df_class, df_karte, selected_student_for_suggestion)
            
            if individual_suggestions:
                for suggestion in individual_suggestions:
                    priority_color = {
                        'high': '🔴',
                        'medium': '🟡',
                        'low': '🟢'
                    }
                    
                    st.markdown(f"""
                    <div style="
                        border-left: 4px solid {'#ff4444' if suggestion['priority'] == 'high' else '#ffaa00' if suggestion['priority'] == 'medium' else '#44ff44'};
                        padding: 15px;
                        margin: 10px 0;
                        background-color: #f8f9fa;
                        border-radius: 5px;
                    ">
                        <h4>{priority_color[suggestion['priority']]} {suggestion['category']}</h4>
                        <p>{suggestion['suggestion']}</p>
                    </div>
                    """, unsafe_allow_html=True)
            else:
                st.info(f"{selected_student_for_suggestion}さんについて、現在特別な提案はありません。")
        
        # カルテ活用の提案
        st.subheader("📝 カルテ活用の提案")
        
        karte_suggestions = []
        
        # カルテが少ない児童の特定
        karte_counts = df_karte['生徒名'].value_counts()
        students_with_few_karte = [s for s in students_in_class if s not in karte_counts.index or karte_counts[s] < 2]
        
        if students_with_few_karte:
            karte_suggestions.append(f"カルテ記録が少ない児童: {', '.join(students_with_few_karte)}。継続的な観察記録を推奨します。")
        
        # 発言はあるがカルテがない児童
        students_speak_no_karte = [s for s in students_in_class if s not in df_karte['生徒名'].values]
        if students_speak_no_karte:
            karte_suggestions.append(f"発言はあるがカルテ記録がない児童: {', '.join(students_speak_no_karte)}。行動観察の記録を開始することを推奨します。")
        
        for suggestion in karte_suggestions:
            st.info(suggestion)

else:
    st.info("授業記録CSVファイルとカルテテキストファイルをアップロードしてください。")
    
    # 使用方法の説明
    st.header("📖 使用方法")
    
    st.markdown("""
    ### このツールの特徴
    
    1. **📝 カルテ履歴の確認**
       - 児童を選択すると、これまでのカルテをページめくり形式で確認できます
       - 日々の気づきを時系列で振り返ることができます
    
    2. **🤝 人間関係の記録・可視化**
       - 児童間の関係（仲良し、対立、教え合いなど）を記録できます
       - ネットワーク図で関係性を一目で確認できます
    
    3. **📊 授業分析**
       - 発言パターンや参加度を分析します
       - 授業の流れを可視化し、改善点を特定します
    
    4. **💡 具体的な提案**
       - データに基づいた具体的な指導提案を提供します
       - 個別児童と授業全体の両方に対応します
    """)
    
    # サンプルデータの表示
    st.header("📄 サンプルデータ形式")
    
    st.subheader("授業記録CSVの例")
    sample_class = pd.DataFrame({
        '発言番号': [1, 2, 3, 4, 5, 6, 7, 8],
        '発言者': ['教師', '伊藤', '教師', '鈴木', '田中', '教師', '伊藤', '鈴木'],
        '発言内容': [
            '今日は三角形について学びます',
            '三角形の内角の和は何度ですか？',
            'いい質問ですね。みなさんはどう思いますか？',
            '180度です！',
            '僕も180度だと思います',
            'その通りです。では、なぜそうなるのでしょうか？',
            '角度を測ってみたからです',
            '証明もできるんじゃないですか？'
        ]
    })
    st.dataframe(sample_class)
    
    st.subheader("カルテテキストの例")
    sample_karte = """#伊藤 積極的に質問する。理解が早い
#鈴木 他の児童の意見をよく聞いている
#田中 発言は少ないが、うなずいて聞いている
#伊藤 今日は特に集中していた
#鈴木 伊藤さんの質問に対して建設的な意見を言えた
#田中 グループ活動では積極的に参加していた"""
    st.code(sample_karte)
