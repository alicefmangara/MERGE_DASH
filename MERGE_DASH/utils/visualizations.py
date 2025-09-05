import base64
import io

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm
from matplotlib.animation import FuncAnimation
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import FancyArrowPatch


def plot_emotional_trajectory(pred_labels):
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
        
        if frame > 0:
            line = ax1.plot(points[:frame+1,0], points[:frame+1,1], 
                          color='gray', linestyle=':', alpha=0.7)[0]
            animated_elements.append(line)
        
        for i in range(frame + 1):
            color = colors[pred_labels[i]]
            point = ax1.scatter(points[i,0], points[i,1], color=color, s=120, zorder=3)
            
            
            x_offset = 0.03 if points[i,0] < 0.5 else -0.03  
            y_offset = 0.03 if points[i,1] < 0.5 else -0.03  
            
            
            ax1.text(points[i,0]+x_offset, points[i,1]+y_offset, str(i+1), 
                    fontsize=10, ha='center', va='center', 
                    color='black', bbox=dict(facecolor='white', alpha=0.8, boxstyle='round,pad=0.2'),
                    zorder=5)  # Garante que fique acima de tudo
            
            # Emoji posicionado no topo do ponto
            emoji = ax1.text(points[i,0], points[i,1]+0.05, emoji_map[pred_labels[i]], 
                           fontsize=20, ha='center', va='center', zorder=4)
            
            animated_elements.extend([point, emoji])
            
            if i > 0:
                
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

    # === 2. Linha do Tempo EmocionaL ===
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

    #
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
    
    # Cria gradiente de cores
    
    cmap = LinearSegmentedColormap.from_list('custom', ['#99c99b', '#de4d4d', '#58add7', '#e8dd63'])
    colors_grad = [cmap(i/len(pred_labels)) for i in range(len(pred_labels))]
    
    # Plota área com gradiente
    x = range(1, len(pred_labels))
    y = np.cumsum(np.sqrt(np.sum(np.diff(points, axis=0)**2, axis=1)))
    ax3.fill_between(x, y, color='#800000', alpha=0.1)
    
    
    line, = ax3.plot(x, y, color='#800000', linewidth=2, alpha=0)  # Invisible line for picking
    for i in range(len(x)):
        ax3.plot(x[i:i+2], y[i:i+2], color=colors[pred_labels[i]], linewidth=2)
        ax3.scatter(x[i], y[i], color=colors[pred_labels[i]], s=80, edgecolor='white', zorder=3)
    
    # Adiciona interatividade (hover)
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
