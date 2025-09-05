import dash
import dash_bootstrap_components as dbc
import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from dash import Input, Output, State, dash_table, dcc, html
from sklearn.metrics import accuracy_score, f1_score
from transformers import RobertaTokenizer, RobertaModel
#from utils.visualizations import plot_emotional_trajectory -> para servidor melhor código das visualizações no mesmo file

import base64
import io

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm
from matplotlib.animation import FuncAnimation
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import FancyArrowPatch


def plot_emotional_trajectory(pred_labels):
    # Configurações iniciais (cores, coordenadas)
    coords = {
        'Q1': (0.75, 0.75), 'Q2': (0.25, 0.75), 
        'Q3': (0.25, 0.25), 'Q4': (0.75, 0.25)
    }
    colors = {
        'Q1': '#99c99b', 'Q2': '#de4d4d',
        'Q3': '#58add7', 'Q4': '#e8dd63'
    }
    emoji_map = {"Q1": "😊", "Q2": "😠", "Q3": "😢", "Q4": "😌"}
    
    points = np.array([coords[q] for q in pred_labels])
    
    fig1, ax1 = plt.subplots(figsize=(8, 8))
    ax1.set_xlim(0, 1)
    ax1.set_ylim(0, 1)
    ax1.set_title("(Russell's Model) Emotional Trajectory", fontsize=14, fontweight='bold', pad=20)
    ax1.set_xlabel("Valence", fontsize=12)
    ax1.set_ylabel("Arousal", fontsize=12)
    ax1.grid(True, alpha=0.3)
    
    # Quadrantes coloridos
    for (xy, width, height), color in zip(
        [((0.5, 0.5), 0.5, 0.5), ((0, 0.5), 0.5, 0.5),
         ((0, 0), 0.5, 0.5), ((0.5, 0), 0.5, 0.5)],
        [colors['Q1'], colors['Q2'], colors['Q3'], colors['Q4']]
    ):
        ax1.add_patch(plt.Rectangle(xy, width, height, color=color, alpha=0.2))
    
    # Labels dos quadrantes
    for label, pos in [('Q1', (0.97, 0.97)), ('Q2', (0.03, 0.97)),
                      ('Q3', (0.03, 0.03)), ('Q4', (0.97, 0.03))]:
        ax1.text(*pos, label, fontsize=14, color='#444', 
                ha='right' if pos[0] > 0.5 else 'left',
                va='top' if pos[1] > 0.5 else 'bottom', alpha=0.6)

    animated_elements = []

    def init():
        scat = ax1.scatter(points[:,0], points[:,1], color='gray', s=50, alpha=0.3)
        animated_elements.append(scat)
        return animated_elements

    def update(frame):
        for element in animated_elements[1:]:
            element.remove()
        animated_elements[1:] = []
        
        # Desenha a linha de trajetória
        if frame > 0:
            line = ax1.plot(points[:frame+1,0], points[:frame+1,1], 
                          color='gray', linestyle=':', alpha=0.7)[0]
            animated_elements.append(line)
        
        for i in range(frame + 1):
            color = colors[pred_labels[i]]
            point = ax1.scatter(points[i,0], points[i,1], color=color, s=120, zorder=3)
            
            # Posicionamento inteligente do número para evitar sobreposição
            # Alterna entre direita e esquerda baseado no quadrante
            x_offset = 0.03 if points[i,0] < 0.5 else -0.03  # Direita para Q2/Q3, esquerda para Q1/Q4
            y_offset = 0.03 if points[i,1] < 0.5 else -0.03  # Acima para Q1/Q2, abaixo para Q3/Q4
            
            # Número do segmento posicionado estrategicamente
            ax1.text(points[i,0]+x_offset, points[i,1]+y_offset, str(i+1), 
                    fontsize=10, ha='center', va='center', 
                    color='black', bbox=dict(facecolor='white', alpha=0.8, boxstyle='round,pad=0.2'),
                    zorder=5)  # Garante que fique acima de tudo
            
            # Emoji posicionado no topo do ponto
            emoji = ax1.text(points[i,0], points[i,1]+0.05, emoji_map[pred_labels[i]], 
                           fontsize=20, ha='center', va='center', zorder=4)
            
            animated_elements.extend([point, emoji])
            
            if i > 0:
                # Seta com cor progressiva e mais fina para melhor visualização
                arrow_color = plt.cm.Reds(0.3 + 0.5*(i/len(pred_labels)))
                arrow = ax1.arrow(
                    points[i-1,0], points[i-1,1],
                    points[i,0]-points[i-1,0], points[i,1]-points[i-1,1],
                    head_width=0.02, head_length=0.04, 
                    fc=arrow_color, ec=arrow_color, alpha=0.8,
                    length_includes_head=True,
                    width=0.002,  
                    zorder=2  
                )
                animated_elements.append(arrow)
        
        return animated_elements

    ani = FuncAnimation(fig1, update, frames=len(pred_labels),
                       init_func=init, interval=1000, blit=False)
    
    gif_path = "emotional_trajectory.gif"
    ani.save(gif_path, writer='pillow', fps=1, dpi=100)
    
    with open(gif_path, "rb") as f:
        gif_data = base64.b64encode(f.read()).decode("utf-8")
    gif_img = f"data:image/gif;base64,{gif_data}"
    
    plt.close(fig1)

    figs = []

    figs.append(fig1)

    # === 2. Linha do Tempo Emocional  ===
    fig2, ax2 = plt.subplots(figsize=(10, 5))

    # Configuração do gráfico
    ax2.set_title("Emotional TimeLine", fontsize=14, fontweight='bold', pad=20)
    ax2.set_xlabel("Segment Number", fontsize=12)
    ax2.set_ylabel("Emotional Quadrant", fontsize=12)
    ax2.set_xticks(range(1, len(pred_labels) + 1))
    ax2.set_yticks([1, 2, 3, 4])
    ax2.set_ylim(0.5, 4.5)
    ax2.grid(True, axis='both', alpha=0.3)

   
    for y, label in zip([1, 2, 3, 4], ['Q1', 'Q2', 'Q3', 'Q4']):
        ax2.text(-0.5, y, emoji_map[label], 
                fontsize=24, 
                ha='center', va='center',
                bbox=dict(boxstyle='round', 
                        facecolor=colors[label], 
                        alpha=0.7, 
                        edgecolor='white'))

    
    ax2.set_yticklabels([])

    
    for i, label in enumerate(pred_labels):
        y_val = int(label[1])  # Extrai o número do quadrante (1-4)
        
        
        ax2.scatter(i+1, y_val, color=colors[label], s=200, zorder=3)
        
        ax2.text(i+1, y_val, emoji_map[label], 
                fontsize=16, ha='center', va='center',
                bbox=dict(facecolor='white', alpha=0.8, boxstyle='round,pad=0.2'))

    for i in range(len(pred_labels)-1):
        ax2.plot([i+1, i+2], 
                [int(pred_labels[i][1]), int(pred_labels[i+1][1])], 
                color=colors[pred_labels[i]], 
                linewidth=2, 
                alpha=0.6)

    annot = ax2.annotate("", xy=(0,0), xytext=(20,20), 
                        textcoords="offset points",
                        bbox=dict(boxstyle="round", fc="white", alpha=0.9),
                        arrowprops=dict(arrowstyle="->"))
    annot.set_visible(False)

    def update_annot(y, label):
        annot.xy = (0, y)
        annot.set_text(f"{label}: {['Alta Valência, Alta Arousal', 'Baixa Valência, Alta Arousal', 'Baixa Valência, Baixa Arousal', 'Alta Valência, Baixa Arousal'][y-1]}")
        annot.get_bbox_patch().set_facecolor(colors[label])
        annot.get_bbox_patch().set_alpha(0.9)

    def hover(event):
        vis = annot.get_visible()
        if event.inaxes == ax2:
            for y, label in zip([1, 2, 3, 4], ['Q1', 'Q2', 'Q3', 'Q4']):
                
                if (event.ydata > y-0.4 and event.ydata < y+0.4 and 
                    event.xdata < 0.5 and event.xdata > -1):
                    update_annot(y, label)
                    annot.set_visible(True)
                    fig2.canvas.draw_idle()
                    return
            if vis:
                annot.set_visible(False)
                fig2.canvas.draw_idle()

    fig2.canvas.mpl_connect("motion_notify_event", hover)
        
    figs.append(fig2)

    # === 3. Variação Emocional Total (gradiente) ===
    fig3, ax3 = plt.subplots(figsize=(8, 3))
    
    # Configuração do gráfico
    ax3.set_title("Variação Emocional Total", fontsize=14, fontweight='bold', pad=20)
    ax3.set_xlabel("Transição entre Estrofes", fontsize=12)
    ax3.set_ylabel("Distância Acumulada", fontsize=12)
    
    
    cmap = LinearSegmentedColormap.from_list('custom', ['#99c99b', '#de4d4d', '#58add7', '#e8dd63'])
    colors_grad = [cmap(i/len(pred_labels)) for i in range(len(pred_labels))]
    
    x = range(1, len(pred_labels))
    y = np.cumsum(np.sqrt(np.sum(np.diff(points, axis=0)**2, axis=1)))
    ax3.fill_between(x, y, color='#800000', alpha=0.1)
    
    line, = ax3.plot(x, y, color='#800000', linewidth=2, alpha=0)  # Invisible line for picking
    for i in range(len(x)):
        ax3.plot(x[i:i+2], y[i:i+2], color=colors[pred_labels[i]], linewidth=2)
        ax3.scatter(x[i], y[i], color=colors[pred_labels[i]], s=80, edgecolor='white', zorder=3)
    
    annot = ax3.annotate("", xy=(0,0), xytext=(20,20), textcoords="offset points",
                        bbox=dict(boxstyle="round", fc="w"),
                        arrowprops=dict(arrowstyle="->"))
    annot.set_visible(False)
    
    def update_annot(ind):
        x_val, y_val = x[ind], y[ind]
        annot.xy = (x_val, y_val)
        text = f"Transição {x_val}\nDistância: {y_val:.2f}"
        annot.set_text(text)
    
    def hover(event):
        vis = annot.get_visible()
        if event.inaxes == ax3:
            cont, ind = line.contains(event)
            if cont:
                update_annot(ind['ind'][0])
                annot.set_visible(True)
                fig3.canvas.draw_idle()
            else:
                if vis:
                    annot.set_visible(False)
                    fig3.canvas.draw_idle()
    
    fig3.canvas.mpl_connect("motion_notify_event", hover)
    
    figs.append(fig3)

    # === Convertendo para imagens base64 ===
    images = []
    for fig in figs:
        buf = io.BytesIO()
        plt.tight_layout()
        fig.savefig(buf, format="png", dpi=100, bbox_inches='tight')
        plt.close(fig)
        buf.seek(0)
        encoded = base64.b64encode(buf.read()).decode("utf-8")
        images.append(f"data:image/png;base64,{encoded}")
    
    # Salvar animação como GIF
    gif_path = "emotional_trajectory.gif"
    ani.save(gif_path, writer='pillow', fps=1, dpi=100)
    
    # Converter GIF para base64
    with open(gif_path, "rb") as f:
        gif_data = base64.b64encode(f.read()).decode("utf-8")
    gif_img = f"data:image/gif;base64,{gif_data}"
    
    return gif_img, images[1] # Retorna tanto o GIF q


# ====== NEW CNN MODEL ARCHITECTURE ======
class CNN(nn.Module):
    def __init__(self, input_type="sequence", num_classes=4, hidden_dim=256):
        super().__init__()
        self.input_type = input_type
        
        if input_type == "sequence":
            # For raw sequences
            self.conv = nn.Sequential(
                nn.Conv1d(768, 256, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.MaxPool1d(2),
                nn.Conv1d(256, 256, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.AdaptiveMaxPool1d(1)
            )
            self.classifier = nn.Linear(256, num_classes)
        else:
            # For pooled embeddings
            self.classifier = nn.Sequential(
                nn.Linear(768, hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.5),
                nn.Linear(hidden_dim, num_classes)
            )
    
    def forward(self, x):
        if self.input_type == "sequence":
            x = x.permute(0, 2, 1)  # [batch, channels, seq_len]
            x = self.conv(x).squeeze(2)
        return self.classifier(x)

# ====== EMOTION CLASSIFIER ======
class EmotionClassifier:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")
        
        # Load the saved CNN model
        model_path = "/home/alicemangara/emotion_classifier/MEVD/delta_emocional/emotion_lyrics_dash/best_model/best_model.pth"
        checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)
        
        # Load label encoder
        self.le = joblib.load("/home/alicemangara/emotion_classifier/MEVD/delta_emocional/emotion_lyrics_dash/best_model/label_encoder.pkl")

        # Initialize and load CNN model
        self.cnn_model = CNN(input_type="sequence", num_classes=len(self.le.classes_))
        self.cnn_model.load_state_dict(checkpoint['model_state_dict'])
        self.cnn_model.to(self.device)
        self.cnn_model.eval()
        
        # Load RoBERTa tokenizer and model
        roberta_path = "/home/alicemangara/emotion_classifier/MEVD/ContextualEmb/results2/finetuned_models2/roberta"
        self.tokenizer = RobertaTokenizer.from_pretrained(roberta_path)
        self.roberta_model = RobertaModel.from_pretrained(roberta_path)
        self.roberta_model.to(self.device)
        self.roberta_model.eval()
    
    def generate_sequences(self, sentences, max_len=50, batch_size=8):
        """Generate RoBERTa embeddings as sequences for CNN input"""
        all_sequences = []
        
        for i in range(0, len(sentences), batch_size):
            batch = sentences[i:i+batch_size]
            inputs = self.tokenizer(
                batch,
                padding='max_length',
                truncation=True,
                max_length=max_len,
                return_tensors="pt"
            ).to(self.device)

            with torch.no_grad():
                outputs = self.roberta_model(**inputs)
                last_hidden = outputs.last_hidden_state
                all_sequences.append(last_hidden.cpu().numpy())
        
        sequences = np.vstack(all_sequences)
        
        # Pad/truncate to max_len
        if sequences.shape[1] < max_len:
            pad_width = [(0, 0), (0, max_len - sequences.shape[1]), (0, 0)]
            sequences = np.pad(sequences, pad_width, mode='constant')
        elif sequences.shape[1] > max_len:
            sequences = sequences[:, :max_len, :]
        
        return sequences
    
    def predict(self, texts):
        """Predict emotions using CNN model"""
        # Generate sequences
        sequences = self.generate_sequences(texts)
        
        # Convert to tensor
        sequences_tensor = torch.FloatTensor(sequences).to(self.device)
        
        # Predict
        with torch.no_grad():
            outputs = self.cnn_model(sequences_tensor)
            predictions = torch.argmax(outputs, dim=1).cpu().numpy()
        
        return self.le.inverse_transform(predictions)

# Initialize classifier
classifier = EmotionClassifier()

# Carregar o dataset consolidado
df_music_dataset = pd.read_csv("/home/alicemangara/emotion_classifier/MEVD/delta_emocional/emotion_lyrics_dash/dash_data_processed.csv")
df_music_dataset["QL"] = df_music_dataset["QL"].astype(str)

# def get_mpnet_embeddings(texts):
#     embeddings = []
#     with torch.no_grad():
#         for text in texts:
#             inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=128)
#             outputs = mpnet(**inputs)
#             pooled = torch.max(outputs.last_hidden_state, dim=1).values
#             embeddings.append(pooled.squeeze().numpy())
#     return np.vstack(embeddings)

def label_to_coords(label):
    coords = {'Q1': (1, 1), 'Q2': (1, -1), 'Q3': (-1, -1), 'Q4': (-1, 1)}
    return coords.get(label, (0, 0))

def get_segment_table(df):
    emoji_map = {"Q1": "😊", "Q2": "😠", "Q3": "😢", "Q4": "😌"}
    df["Emoji"] = df["Label"].map(emoji_map)

    return dash_table.DataTable(
        columns=[
            {"name": "ID", "id": "IDSentence"},
            {"name": "Segment", "id": "Sentence"},
            {"name": "Quadrant", "id": "Label"},
            {"name": "True Label", "id": "TrueLabelDisplay"},
            {"name": "Emoji", "id": "Emoji"}
        ],
        data=df.to_dict("records"),
        style_table={"overflowX": "auto", "height": "100%", "overflowY": "auto"},
        style_header={"backgroundColor": "black", "color": "white", "fontWeight": "bold"},
        style_cell={"textAlign": "left", "padding": "6px"},
        style_data_conditional=[
            {"if": {"filter_query": '{Label} = "Q1"'}, "backgroundColor": '#99c99b'},
            {"if": {"filter_query": '{Label} = "Q2"'}, "backgroundColor": '#de4d4d'},
            {"if": {"filter_query": '{Label} = "Q3"'}, "backgroundColor": '#58add7'},
            {"if": {"filter_query": '{Label} = "Q4"'}, "backgroundColor": '#e8dd63'},
            
            # destacar TrueLabel quando houver erro
            {
                "if": {
                    "filter_query": '{Label} != {TrueLabel}',
                    "column_id": "TrueLabelDisplay"
                },
                "color": "red",
                "fontWeight": "bold"
            },
        ],
        style_cell_conditional=[
            {
                "if": {"column_id": "Sentence"},
                "whiteSpace": "pre-wrap",
                "maxWidth": "400px",
                "overflow": "hidden",
                "textOverflow": "ellipsis"
            }
        ]
    )

app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
server = app.server

app.layout = html.Div([
    html.Div([
        html.H1([
    html.Span("MERGE", className="merge-title", **{"data-tooltip": "Music Emotion Recognition: Next Generation"}),
    html.Span(" ", style={"margin": "0 10px"}),
    html.Span("LMEVD", className="lmevd-title", **{"data-tooltip": "Lyrics Music Emotion Variation Detection"}),
    html.Span(" ♫", className="black-note")
], className="main-title"),

        html.Img(src="/assets/image-removebg-preview.png", className="header-icon"),
        html.Div([
            html.Span("♪", className="note note1"),
            html.Span("♫", className="note note2"),
            html.Span("♬", className="note note3"),
        ], className="music-notes")
    ], className="header-bar"),

    html.Div([
            html.Div([
                html.Div([
                    dbc.Button("Select Music", id="open_music_modal", className="lyrics-dropdown", color="dark"),
                    dcc.Store(id="music_selector"),  # Armazena música selecionada
                    html.Div(id="selected_music_display", style={"marginTop": "10px", "fontWeight": "bold"}),

                    dbc.Modal([
                        dbc.ModalHeader("Select a Music Lyric"),
                        dbc.ModalBody([
                            dcc.Input(id="search_music_input", type="text", placeholder="Search...", debounce=True, className="search-box"),
                            html.Div(id="music_list_scroll", className="music-list-scroll")
                        ]),
                        dbc.ModalFooter(
                            dbc.Button("Close", id="close_music_modal", className="ms-auto", n_clicks=0)
                        )
                    ], id="music_modal", is_open=False, size="lg", scrollable=True, backdrop="static"),
                ], className="selector-container"),
            dbc.Button(
                "Analyze Emotion",
                id="submit_button",
                color="danger",
                className="analyze-button"
            ),
            html.Div(id="lyrics_output", className="lyrics-box")
        ], className="left"),
        html.Div([
            html.H5("Segments", className="segment-title"),
            html.Div(id="segment_table"),
            html.Div(id="evaluation_metrics_div", style={"marginTop": "15px", "textAlign": "center"})
        ], className="center"),
        html.Div([
            html.H5("Lyrics Music Emotion Variation", className="graph-title"),
            html.Div(id="emotional_graphs")
        ], className="right"),
    ], className="content")
])
@app.callback(
    Output("lyrics_output", "children"),
    Output("segment_table", "children"),
    Output("evaluation_metrics_div", "children"), 
    Output("emotional_graphs", "children"),
    Input("submit_button", "n_clicks"),
    State("music_selector", "data"),

)

def predict_emotions(n_clicks, selected_music):
    if not n_clicks or not selected_music:
        return "", "", "", ""

    # Get both raw and cleaned lyrics
    song_df = df_music_dataset[df_music_dataset["Music"] == selected_music]
    song_df = song_df.sort_values('Segment_Position')
    raw_lyrics = song_df["Lyrics"].tolist()  # For display
    cleaned_lyrics = song_df["Cleaned_Lyrics"].tolist()  # For model
    true_labels = song_df["QL"].apply(lambda x: f"Q{int(float(x))}").tolist()

    # Use CLEANED lyrics for prediction with CNN model
    preds_labels = classifier.predict(cleaned_lyrics) 

    full_lyrics = "\n".join(song_df["Lyrics"])

    #DataFrame com resultados
    data = []
    for i, (raw_text, label, true_label) in enumerate(zip(raw_lyrics, preds_labels, true_labels), start=1):
        valence, arousal = label_to_coords(label)
        data.append({
            "IDSentence": i,
            "Sentence": raw_text,
            "Label": label,
            "TrueLabel": true_label,
            "Valence": valence,
            "Arousal": arousal
        })

    df = pd.DataFrame(data)
    df["TrueLabel"] = true_labels

    # Adiciona coluna com ícone de alerta se estiver errado
    df["TrueLabelDisplay"] = [
        f"{true} ⚠️" if pred != true else true
        for pred, true in zip(df["Label"], df["TrueLabel"])
    ]

    gif_img, timeline_img = plot_emotional_trajectory(preds_labels)
    
    
    # graph group layout
    graph_group = html.Div([
    html.Div([
        html.Img(src=timeline_img, className="timeline-plot"),
        html.Div("Emotional Timeline", className="plot-tooltip")
    ], className="timeline-container"),
    
    html.Div([
        html.Img(src=gif_img, className="animated-trajectory"),
        html.Div("Emotional Trajectory", className="plot-tooltip")
    ], className="plot-container", style={"width": "100%"})  
], className="right-graphs-container")

    
    label_to_class = {'Q1': 'segment-q1', 'Q2': 'segment-q2', 'Q3': 'segment-q3', 'Q4': 'segment-q4'}
    lyrics_text_with_highlight = []
    offset = 0
    full_lyrics = "\n\n".join(raw_lyrics)

    for raw_segment, label in zip(raw_lyrics, preds_labels):
        index = full_lyrics.find(raw_segment, offset)
        if index > offset:
            lyrics_text_with_highlight.append(full_lyrics[offset:index])
        css_class = f"segment-box {label_to_class.get(label, '')}"
        lyrics_text_with_highlight.append(html.Div(raw_segment, className=css_class))
        offset = index + len(raw_segment)
    if offset < len(full_lyrics):
        lyrics_text_with_highlight.append(full_lyrics[offset:])

    lyrics_divs = html.Div(lyrics_text_with_highlight, className="lyrics-full-text")
    table = get_segment_table(df)
    gif_img, timeline_img= plot_emotional_trajectory(preds_labels)

    print("True labels:", true_labels)
    print("Predicted labels:", preds_labels)
    print("Unique true labels:", set(true_labels))
    print("Unique predicted labels:", set(preds_labels))
    print(f"Length true: {len(true_labels)}, Length pred: {len(preds_labels)}")

    try:
        f1 = f1_score(true_labels, preds_labels, average="macro")
        acc = accuracy_score(true_labels, preds_labels)
        evaluation_text = html.Div([
            html.H5("Model Evaluation", className="metric-title"),
            html.P(f"F1-score (macro): {f1:.2f}"),
            html.P(f"Accuracy: {acc:.2f}")
        ], className="evaluation-metrics", style={"textAlign": "center", "marginTop": "10px"})
    except Exception:
        evaluation_text = html.Div()  # vazio se erro

    return lyrics_divs, table, evaluation_text, graph_group

@app.callback(
    Output("selected_music_display", "children"),
    Output("music_selector", "data"),
    Output("music_modal", "is_open"),
    Input("open_music_modal", "n_clicks"),
    Input("close_music_modal", "n_clicks"),
    Input({'type': 'music_item', 'index': dash.ALL}, "n_clicks"),
    State("music_list_scroll", "children"),
    State("music_modal", "is_open"),
    prevent_initial_call=True
)
def handle_music_modal(open_click, close_click, n_clicks_list, children, is_open):
    ctx = dash.callback_context

    if not ctx.triggered:
        return dash.no_update, dash.no_update, dash.no_update

    triggered_id = ctx.triggered[0]["prop_id"].split(".")[0]

    # Clicar em item da lista de músicas
    if "music_item" in triggered_id:
        selected_idx = eval(triggered_id)["index"]
        selected_music = children[selected_idx]["props"]["children"]
        return f"Selected: {selected_music}", selected_music, False

    # Abrir ou fechar modal
    if "open_music_modal" in triggered_id or "close_music_modal" in triggered_id:
        return dash.no_update, dash.no_update, not is_open

    return dash.no_update, dash.no_update, is_open

    
@app.callback(
    Output("music_list_scroll", "children"),
    Input("search_music_input", "value")
)
def update_music_list(search_value):
    filtered = sorted(df_music_dataset["Music"].unique())
    if search_value:
        filtered = [m for m in filtered if search_value.lower() in m.lower()]
    return [
        html.Div(m, className="music-item", n_clicks=0, id={'type': 'music_item', 'index': i})
        for i, m in enumerate(filtered)
    ]


if __name__ == "__main__":
    app.run(debug=True)

