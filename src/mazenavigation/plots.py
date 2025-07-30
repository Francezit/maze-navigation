import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Patch
from matplotlib.lines import Line2D
from collections import defaultdict
import numpy as np

from .exploration import ExplorationResultCollection
from .maze import Maze


def plot_multi_agent_exploration(maze: Maze, filename: str, results: ExplorationResultCollection):
    assert maze.is_loaded, "Maze is not loaded yet"

    fig, ax = plt.subplots(figsize=(10, 10))

    ax.hist(maze.weights)
    ax.set_xlabel("Weights")
    ax.set_ylabel("Counts")
    fig.savefig(filename)
    plt.close(fig)

    # Converti la collezione in lista se necessario
    if isinstance(results, ExplorationResultCollection):
        results_list = list(results)
    else:
        results_list = results

    # Configurazione figura
    fig, ax = plt.subplots(figsize=(14, 10))
    plt.subplots_adjust(right=0.6)

    # Disegno labirinto
    ax.imshow(np.floor(maze.matrix), cmap=plt.cm.binary,
              interpolation='nearest')

    # Calcolo statistiche
    total = len(results_list)
    dead = sum(1 for r in results_list if r.is_killed)
    survived = total - dead
    cmap = plt.cm.get_cmap("tab20", total)

    # Disegno percorsi
    for i, result in enumerate(results_list):
        if result.path:
            x = [p[1] for p in result.path]
            y = [p[0] for p in result.path]
            ax.plot(x, y, color=cmap(i), linewidth=2,
                    linestyle='--' if result.is_killed else '-')

    # Punti chiave
    ax.add_patch(
        Circle((maze.enter_point[1], maze.enter_point[0]), 0.5, color='lime'))
    ax.add_patch(
        Circle((maze.exit_point[1], maze.exit_point[0]), 0.5, color='orange'))

    # Blocco informazioni unificato
    info_text = [
        "📊 AGENT STATISTICS",
        f"🧑 Total: {total}",
        f"✅ Survived: {survived}",
        f"❌ Dead: {dead}",
        f"⚡ Rate: {survived/total:.0%}",
        "",
        "🎨 AGENT COLORS:"
    ]

    # Aggiungi righe colorate per ogni agente
    # Ora può usare l'indicizzazione
    for i, agent in enumerate(results_list[:12]):
        status = "❌" if agent.is_killed else "✅"
        agent_name = agent.agent_name[:15] + \
            "..." if len(agent.agent_name) > 15 else agent.agent_name
        info_text.append(f"  {status} {cmap(i)[:3]} {agent_name}")

    if total > 12:
        info_text.append(f"...plus {total-12} more agents")

    fig.text(0.65, 0.5, "\n".join(info_text),
             ha='left', va='center', fontsize=9,
             bbox=dict(facecolor='white', alpha=0.9, boxstyle='round'))

    ax.set_xticks([])
    ax.set_yticks([])
    plt.savefig(filename, bbox_inches='tight', dpi=150)
    plt.close()


def plot_multi_agent_exploration_statistics(maze: Maze, filename: str, results: ExplorationResultCollection, show_weights=False):
    """Draw all agent paths on the maze with advanced statistics."""
    assert maze.is_loaded, "Maze is not loaded yet"

    matrix = maze.matrix
    enter_maze = maze.enter_point
    exit_maze = maze.exit_point

    # Configure figure with space for legend
    fig, ax = plt.subplots(figsize=(16, 12))
    plt.subplots_adjust(right=0.7, left=0.05, top=0.95, bottom=0.05)

    # Maze background
    if show_weights:
        ax.imshow(matrix, cmap=plt.cm.binary, interpolation='nearest')
    else:
        ax.imshow(np.floor(matrix), cmap=plt.cm.binary, interpolation='nearest')

    # Calculate statistics
    total_agents = len(results)
    dead_agents = sum(1 for r in results if r.is_killed)
    survived_agents = total_agents - dead_agents
    survival_rate = survived_agents/total_agents
    agent_names = [r.agent_name for r in results]

    # Prepare path data
    all_paths = []
    death_positions = defaultdict(int)
    cmap = plt.cm.get_cmap("tab20", len(results))

    # First pass: collect path data
    for i, result in enumerate(results):
        path = result.path
        if path:
            x_coords = [x[1] for x in path]
            y_coords = [y[0] for y in path]
            all_paths.append((x_coords, y_coords, i, result.is_killed))
            if result.is_killed:
                death_positions[path[-1]] += 1

    # Second pass: draw paths with overlap detection
    path_counts = defaultdict(int)
    for x_coords, y_coords, i, is_dead in all_paths:
        for x, y in zip(x_coords, y_coords):
            path_counts[(x, y)] += 1

    for x_coords, y_coords, i, is_dead in all_paths:
        max_count = max(path_counts[(x, y)]
                        for x, y in zip(x_coords, y_coords))
        line_style = (0, (5, 2)) if max_count > 3 else (
            '--' if is_dead else '-')

        ax.plot(x_coords, y_coords,
                color=cmap(i),
                linewidth=1.5 if max_count > 3 else 2,
                linestyle=line_style,
                alpha=0.8 if max_count > 3 else 1.0)

    # Key points
    ax.add_patch(
        Circle((enter_maze[1], enter_maze[0]), 0.5, color='lime', zorder=10))
    ax.add_patch(
        Circle((exit_maze[1], exit_maze[0]), 0.5, color='orange', zorder=10))

    # Death markers
    for pos, count in death_positions.items():
        x, y = pos[1], pos[0]
        ax.add_patch(Circle((x, y), 0.3, color='red', zorder=10))
        if count > 1:
            ax.text(x, y, str(count), color='white', fontsize=10,
                    ha='center', va='center', fontweight='bold', zorder=11)

    # Enhanced legend with statistics and agent list
    legend_elements = [
        Patch(facecolor='lime', edgecolor='black', label='Start'),
        Patch(facecolor='orange', edgecolor='black', label='Exit'),
        Patch(facecolor='red', edgecolor='black', label='Death'),
        Line2D([0], [0], color='black', lw=2, linestyle='-', label='Survived'),
        Line2D([0], [0], color='black', lw=2, linestyle='--', label='Dead'),
        Line2D([0], [0], color='black', lw=2, linestyle=(
            0, (5, 2)), label='Heavy Traffic'),
        Patch(facecolor='none', edgecolor='none', label='\nSTATISTICS'),
        Patch(facecolor='none', edgecolor='none',
              label=f'TOTAL: {total_agents}'),
        Patch(facecolor='none', edgecolor='none',
              label=f'SURVIVED: {survived_agents}'),
        Patch(facecolor='none', edgecolor='none',
              label=f'DEAD: {dead_agents}'),
        Patch(facecolor='none', edgecolor='none',
              label=f'SURVIVAL RATE: {survival_rate:}'),
        Patch(facecolor='none', edgecolor='none', label='\nAGENT COLORS:')
    ]

    # Add agent color patches (max 15 to avoid overcrowding)
    max_agents_to_show = 15
    for i, name in enumerate(agent_names[:max_agents_to_show]):
        legend_elements.append(
            Patch(facecolor=cmap(i), edgecolor='black', label=name))

    if len(agent_names) > max_agents_to_show:
        legend_elements.append(Patch(facecolor='none', edgecolor='none',
                                     label=f'... +{len(agent_names)-max_agents_to_show} more'))

    ax.legend(handles=legend_elements,
              loc='center left',
              bbox_to_anchor=(1.02, 0.8),
              fontsize=8,
              title="MAZE VISUALIZATION",
              title_fontsize=10)

    ax.set_xticks([])
    ax.set_yticks([])
    ax.grid(False)

    # Save high quality image
    fig.savefig(filename, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close(fig)


__all__ = [
    "plot_multi_agent_exploration",
    "plot_multi_agent_exploration_statistics"
]
