import matplotlib
matplotlib.use("Agg")

from flask import Flask, render_template, request
import networkx as nx
import matplotlib.pyplot as plt
import heapq
import io
import base64
import os

app = Flask(__name__)

default_edges = [
    ('A','B',4),
    ('A','C',2),
    ('B','C',1),
    ('B','D',5),
    ('C','D',8),
    ('C','E',10),
    ('D','E',2),
    ('D','F',6),
    ('E','F',3)
]

heuristic = {}

def calculate_heuristic(graph, goal):

    h = {}

    for node in graph.nodes:

        try:
            dist = nx.shortest_path_length(graph,node,goal,weight='weight')
            h[node] = dist
        except:
            h[node] = 999

    return h


def parse_edges(text):

    edges=[]
    lines=text.strip().split("\n")

    for line in lines:

        parts=line.split()

        if len(parts)==3:

            u,v,w=parts
            edges.append((u,v,int(w)))

    return edges


def a_star(graph,start,goal):

    pq=[]
    heapq.heappush(pq,(0,start,[start]))

    visited=set()

    while pq:

        cost,node,path=heapq.heappop(pq)

        if node==goal:
            return path

        if node in visited:
            continue

        visited.add(node)

        for n in graph.neighbors(node):

            weight=graph[node][n]['weight']

            heapq.heappush(
                pq,
                (cost+weight+heuristic.get(n,0),n,path+[n])
            )

    return None


def greedy(graph,start,goal):

    pq=[(0,start,[start])]
    visited=set()

    while pq:

        pq.sort()

        _,node,path=pq.pop(0)

        if node==goal:
            return path

        visited.add(node)

        for n in graph.neighbors(node):

            if n not in visited:

                pq.append((heuristic.get(n,0),n,path+[n]))

    return None


def draw_graph(graph, path=None):

    plt.figure(figsize=(6,5))

    pos = nx.spring_layout(graph, seed=2)

    # Node labels with heuristic
    labels = {}
    for node in graph.nodes:
        h = heuristic.get(node, "?")
        labels[node] = f"{node}\n(h={h})"

    nx.draw(
        graph,
        pos,
        labels=labels,
        node_color="#8ecae6",
        node_size=2000,
        font_size=10
    )

    # Draw edge weights
    edge_labels = nx.get_edge_attributes(graph, 'weight')
    nx.draw_networkx_edge_labels(graph, pos, edge_labels=edge_labels)

    # Highlight path with arrows
    if path:

        path_edges = list(zip(path, path[1:]))

        nx.draw_networkx_edges(
            graph,
            pos,
            edgelist=path_edges,
            edge_color="red",
            width=3,
            arrows=True,
            arrowstyle='-|>',
            arrowsize=15,
            min_source_margin=25,
            min_target_margin=25
        )

    img = io.BytesIO()
    plt.savefig(img, format="png")
    img.seek(0)
    plt.close()

    return base64.b64encode(img.getvalue()).decode()
@app.route("/",methods=["GET","POST"])
def index():

    result=None
    cost=None
    graph_img=None
    table=[]

    mode="default"
    edges_text=""
    start=""
    goal=""

    edges=default_edges

    if request.method=="POST":

        action=request.form.get("action")
        mode=request.form.get("mode","default")

        edges_text=request.form.get("edges","")

        start=request.form.get("start","")
        goal=request.form.get("goal","")

        algo=request.form.get("algorithm","astar")

        if mode=="custom" and edges_text.strip():
            edges=parse_edges(edges_text)

        G=nx.Graph()
        G.add_weighted_edges_from(edges)

        path=None

        if action=="find":

            if start not in G.nodes or goal not in G.nodes:

                result="Start or Goal node not in graph"

            else:

                global heuristic
                heuristic = calculate_heuristic(G,goal)

                if algo=="astar":
                    path=a_star(G,start,goal)
                else:
                    path=greedy(G,start,goal)

                if path:

                    cost=nx.path_weight(G,path,weight="weight")
                    result=path

                    cumulative=0

                    for i in range(len(path)-1):

                        u=path[i]
                        v=path[i+1]

                        w=G[u][v]['weight']

                        cumulative+=w

                        table.append({
                            "from":u,
                            "to":v,
                            "cost":w,
                            "cumulative":cumulative
                        })

                else:
                    result="No Path Found"

        graph_img=draw_graph(G,path)

    else:

        G=nx.Graph()
        G.add_weighted_edges_from(default_edges)
        graph_img=draw_graph(G)

    return render_template(
        "index.html",
        graph=graph_img,
        result=result,
        cost=cost,
        table=table,
        heuristic=heuristic,
        mode=mode,
        edges=edges_text,
        start=start,
        goal=goal
    )


if __name__=="__main__":

    port=int(os.environ.get("PORT",5000))
    app.run(debug=True)