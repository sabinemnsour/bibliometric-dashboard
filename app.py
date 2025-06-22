# Dash & composants
import dash
from dash import Dash, html, dcc, Input, Output, State, callback
from dash import dash_table
from dash.dependencies import Input, Output
import dash_cytoscape as cyto

# Traitement des données
import pandas as pd
import numpy as np
import ast
import re
from collections import Counter, defaultdict
from itertools import combinations

# Visualisation
import plotly.express as px
import plotly.graph_objects as go
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from pyvis.network import Network

# Graphes & Réseaux
import networkx as nx
import community as community_louvain

# NLP (traitement du texte)
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
nltk.download('punkt')
nltk.download('stopwords')

# Utilitaires
import base64
from io import BytesIO
import tempfile
import os

from nltk.tokenize import TreebankWordTokenizer









# Data
df = pd.read_csv("publications_with_I2M_authors_only.csv")
preview_df = df

# I2M
group_descriptions = {
    "AA – Applied Analysis": (
        "The Applied Analysis group spans a wide spectrum of research — from the theoretical study of partial differential equations (PDEs) "
        "to numerical analysis and interdisciplinary modeling."
    ),
    "AGT – Analysis, Geometry, Topology": (
        "Structured into four main themes, AGT explores complex and harmonic analysis, real and complex geometry, topology, and singularities. "
        "This group delves deep into the abstract mathematical structures shaping space and form."
    ),
    "GDAC – Geometry, Dynamics, Arithmetic, Combinatorics": (
        "GDAC thrives on blurred boundaries — its strength lies in the interaction between its four main themes. "
        "Its scientific life is rich with joint publications and cross-disciplinary collaboration."
    ),
    "AGLR – Arithmetic, Geometry, Logic and Representations": (
        "AGLR brings together three research teams connected by a strong algebraic perspective. "
        "It unites discrete mathematics, number theory, and logical structures through a common lens of representation."
    ),
    "ALEA – Mathematics of Randomness": (
        "The ALEA group functions like a small independent lab, composed of four teams: Probability, Statistics, Signal & Image, and BioMath. "
        "All are united by their use or study of randomness in scientific problems."
    )
}


#Publications per year
pubs_per_year = df['Publication Date'].value_counts().sort_index()
pubs_per_year_df = pubs_per_year.reset_index()
pubs_per_year_df.columns = ['Year', 'Number of Publications']

bar_fig = px.bar(
    pubs_per_year_df,
    x='Year',
    y=[0]*len(pubs_per_year_df),  
    color_discrete_sequence=['#16a085'],
    labels={'Year': 'Year', 'Number of Publications': 'Number of Publications'},
    title='Number of Publications per Year'
)

bar_fig.update_layout(
    plot_bgcolor='white',
    paper_bgcolor='white',
    title_font_size=20,
    margin=dict(t=60, b=40, l=20, r=20),
    xaxis_tickangle=45
)


#  Document Type Distribution
doc_type_counts = df['Document Type'].value_counts()
doc_type_percents = (doc_type_counts / doc_type_counts.sum() * 100).round(1)

doc_type_df = pd.DataFrame({
    'Document Type': doc_type_counts.index,
    'Count': doc_type_counts.values,
    'Percent': doc_type_percents.values
})
doc_type_df['Label'] = doc_type_df.apply(lambda row: f"{row['Count']} ({row['Percent']}%)", axis=1)

doc_bar_fig = go.Figure(
    data=[go.Bar(
        x=[0]*len(doc_type_df),
        y=doc_type_df['Document Type'],
        orientation='h',
        marker_color='#d98880',
        hovertemplate='Document Type: %{y}<br>Count: %{x} (%{customdata}%)<extra></extra>',
        customdata=doc_type_df['Percent']
    )],
    layout=go.Layout(
        title='Distribution of Document Types',
        xaxis=dict(title='Count'),
        yaxis=dict(title='Document Type', categoryorder='total ascending'),
        plot_bgcolor='white',
        paper_bgcolor='white',
        title_font_size=20,
        margin=dict(t=60, b=40, l=20, r=20)
    )
)


# Publications by Research Group

df['Research Groups'] = df['Research Groups'].apply(ast.literal_eval)
df_exploded = df.explode('Research Groups')

group_counts = df_exploded['Research Groups'].value_counts()
group_df = pd.DataFrame({
    'Research Group': group_counts.index,
    'Number of Publications': group_counts.values
})

group_bar_fig = go.Figure(
    data=[go.Bar(
        x=[0]*len(group_df),
        y=group_df['Research Group'],
        orientation='h',
        marker_color='#a569bd',
        hovertemplate='Group: %{y}<br>Publications: %{x}<extra></extra>'
    )],
    layout=go.Layout(
        title="Number of Publications by Research Group",
        xaxis=dict(title='Number of Publications', range=[0, group_df['Number of Publications'].max() + 50]),
        yaxis=dict(title='Research Group', categoryorder='total ascending'),
        plot_bgcolor='white',
        paper_bgcolor='white',
        title_font_size=20,
        margin=dict(t=60, b=40, l=20, r=20)
    )
)


# Publications by Author

df['I2M Authors'] = df['I2M Authors'].apply(ast.literal_eval)
df_authors_exploded = df.explode('I2M Authors')

author_counts = df_authors_exploded['I2M Authors'].value_counts().head(20)
author_df = pd.DataFrame({
    'Author': author_counts.index,
    'Number of Publications': author_counts.values
})

author_fig = go.Figure(
    data=[go.Bar(
        x=[0]*len(author_df),
        y=author_df['Author'],
        orientation='h',
        marker_color='#5dade2',
        hovertemplate='Author: %{y}<br>Publications: %{x}<extra></extra>'
    )],
    layout=go.Layout(
        title='Top 20 Most Prolific I2M Authors',
        xaxis=dict(title='Number of Publications', range=[0, author_df['Number of Publications'].max() + 10]),
        yaxis=dict(title='Author', categoryorder='total ascending'),
        plot_bgcolor='white',
        paper_bgcolor='white',
        title_font_size=20,
        margin=dict(t=60, b=40, l=20, r=20)
    )
)


# Internal vs External Collaborations

import ast
import plotly.graph_objects as go
import numpy as np

def classify_collaboration(i2m_authors, co_authors):
    if not isinstance(i2m_authors, list):
        i2m_authors = []
    if not isinstance(co_authors, list):
        co_authors = []

    if len(i2m_authors) >= 2 and len(co_authors) == 0:
        return 'Internal Only'
    elif len(i2m_authors) >= 2 and len(co_authors) >= 1:
        return 'Internal + External'
    elif len(i2m_authors) == 1 and len(co_authors) >= 1:
        return 'External Only'
    elif len(i2m_authors) == 1 and len(co_authors) == 0:
        return 'Unclassified'
    else:
        return 'Unclassified'

df['I2M Authors'] = df['I2M Authors'].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else x)
df['External Authors'] = df['External Authors'].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else x)
df['Collab Type'] = df.apply(lambda row: classify_collaboration(row['I2M Authors'], row['External Authors']), axis=1)

collab_counts = df['Collab Type'].value_counts().reset_index()
collab_counts.columns = ['Collaboration Type', 'Count']
collab_counts['Percentage'] = (collab_counts['Count'] / collab_counts['Count'].sum() * 100).round(1)
collab_counts['Custom Label'] = collab_counts.apply(
    lambda row: f"{row['Collaboration Type']}<br>Count: {row['Count']}<br>Percent: {row['Percentage']}%",
    axis=1
)

labels = collab_counts['Collaboration Type'].tolist()
values = collab_counts['Count'].tolist()
hovertexts = collab_counts['Custom Label'].tolist()
total = sum(values)
angles = [v / total * 360 for v in values]
color_sequence = ['#aed6f1', '#d2b4de', '#f5b7b1', '#f9e79f']

frames = []
step = 10
theta_cumulative = 0

for k in range(1, step + 1):
    theta_list = []
    width_list = []
    r_list = []
    hovertext_list = []
    color_list = []

    for i in range(len(values)):
        current_theta = angles[i] * (k / step)
        theta = theta_cumulative + current_theta / 2
        theta_list.append(theta)
        width_list.append(current_theta)
        r_list.append(1)
        hovertext_list.append(hovertexts[i])
        color_list.append(color_sequence[i % len(color_sequence)])
        theta_cumulative += angles[i]

    theta_cumulative = 0  

    frames.append(go.Frame(
        data=[go.Barpolar(
            r=r_list,
            theta=theta_list,
            width=width_list,
            marker_color=color_list,
            marker_line_color='white',
            marker_line_width=2,
            opacity=1,
            hovertext=hovertext_list,
            hoverinfo='text',
            name=''  
        )],
        name=str(k)
    ))

collab_fig = go.Figure(
    data=[go.Barpolar(
        r=[], theta=[], width=[],
        marker_color=[],
        marker_line_color='white',
        marker_line_width=2,
        opacity=1,
        hovertext=[],
        hoverinfo='text'
    )],
    layout=go.Layout(
        title="Internal vs External Collaborations (Animated Circle)",
        polar=dict(
            radialaxis=dict(visible=False),
            angularaxis=dict(showticklabels=False, ticks='')
        ),
        showlegend=False,  
        plot_bgcolor='white',
        paper_bgcolor='white',
        title_font_size=20,
        margin=dict(t=60, b=40, l=20, r=20),
        updatemenus=[{
            'type': 'buttons',
            'showactive': False,
            'buttons': [{
                'label': 'Animate',
                'method': 'animate',
                'args': [None, {
                    'frame': {'duration': 250, 'redraw': True},
                    'transition': {'duration': 150, 'easing': 'cubic-in-out'},
                    'fromcurrent': True,
                    'mode': 'immediate'
                }]
            }]
        }]
    ),
    frames=frames
)





# Collaboration Intensity: Internal vs External
df['Num I2M Authors'] = df['I2M Authors'].apply(lambda x: len(x) if isinstance(x, list) else 0)
df['Num External Authors'] = df['External Authors'].apply(lambda x: len(x) if isinstance(x, list) else 0)

def collaboration_type(row):
    if row['Num I2M Authors'] > 0 and row['Num External Authors'] > 0:
        return 'Mixed'
    elif row['Num I2M Authors'] > 0:
        return 'Internal Only'
    elif row['Num External Authors'] > 0:
        return 'External Only'
    else:
        return 'Unknown'

df['Collab Category'] = df.apply(collaboration_type, axis=1)

collab_stats = df['Collab Category'].value_counts().reset_index()
collab_stats.columns = ['Collaboration Type', 'Number of Publications']

total = collab_stats['Number of Publications'].sum()
collab_stats['Percent'] = (collab_stats['Number of Publications'] / total * 100).round(1)

colors = ['#f9c3d4', '#bb8fce', '#aed6f1', '#f7dc6f']

collab_bar_fig = go.Figure(
    data=[go.Bar(
        x=collab_stats['Collaboration Type'],
        y=[0] * len(collab_stats),
        marker_color=colors,
        customdata=collab_stats['Percent'],
        hovertemplate='Type: %{x}<br>Publications: %{y} (%{customdata}%)<extra></extra>'
    )],
    layout=go.Layout(
        title='Collaboration Types Across Publications',
        xaxis=dict(title='Collaboration Type'),
        yaxis=dict(title='Number of Publications', range=[0, collab_stats['Number of Publications'].max() + 50]),
        plot_bgcolor='white',
        paper_bgcolor='white',
        title_font_size=20,
        margin=dict(t=60, b=40, l=20, r=20)
    )
)



# 2.1 Collaboration Between Research Groups
# Co-publication Matrix (Group to Group)



group_pairs = []

for groups in df['Research Groups']:
    if isinstance(groups, list) and len(groups) > 1:
        pairs = combinations(sorted(set(groups)), 2)
        group_pairs.extend(pairs)

pair_counts = Counter(group_pairs)

group_edges_df = pd.DataFrame(pair_counts.items(), columns=['Group Pair', 'Weight'])
group_edges_df[['Group 1', 'Group 2']] = pd.DataFrame(group_edges_df['Group Pair'].tolist(), index=group_edges_df.index)
group_edges_df = group_edges_df[['Group 1', 'Group 2', 'Weight']]

unique_groups = sorted(set(group_edges_df['Group 1']).union(set(group_edges_df['Group 2'])))

#  Network Visualization (NetworkX)
def generate_network_elements():
    elements = []
    nodes = list(set(group_edges_df['Group 1']).union(set(group_edges_df['Group 2'])))
    for node in nodes:
        elements.append({'data': {'id': node, 'label': node}})
    for _, row in group_edges_df.iterrows():
        elements.append({
            'data': {
                'source': row['Group 1'],
                'target': row['Group 2'],
                'weight': row['Weight'],
                'tooltip': row['Weight']
            }
        })
    return elements


# Global Network Metrics
G_groups = nx.Graph()
for _, row in group_edges_df.iterrows():
    G_groups.add_edge(row['Group 1'], row['Group 2'], weight=row['Weight'])

num_nodes = G_groups.number_of_nodes()
num_edges = G_groups.number_of_edges()
average_degree = sum(dict(G_groups.degree()).values()) / num_nodes
density = nx.density(G_groups)
num_connected_components = nx.number_connected_components(G_groups)

largest_cc = max(nx.connected_components(G_groups), key=len)
G_largest_cc = G_groups.subgraph(largest_cc)
diameter = nx.diameter(G_largest_cc)

global_metrics = {
    "Number of Nodes (Research Groups)": f"{num_nodes}",
    "Number of Edges (Collaborations)": f"{num_edges}",
    "Average Degree": f"{average_degree:.2f}",
    "Density": f"{density:.4f}",
    "Number of Connected Components": f"{num_connected_components}",
    "Diameter of Largest Connected Component": f"{diameter}"
}

global_metric_explanations = {
    "Number of Nodes (Research Groups)": "This metric counts how many distinct I2M research groups are represented in the internal collaboration network. A higher number means a wider scientific ecosystem within the institute.",
    
    "Number of Edges (Collaborations)": "This measures how many group pairs collaborated at least once. A high edge count indicates a dense web of collaboration among research groups.",
    
    "Average Degree": "This shows the average number of groups each research group collaborates with. A higher average suggests widespread, inclusive teamwork across I2M.",
    
    "Density": "The density measures how many links exist compared to the maximum possible. A value of 1.0 means a fully connected graph — everyone works with everyone.",
    
    "Number of Connected Components": "This tells us how many isolated collaboration clusters exist. Ideally, there should be only one connected component — indicating cohesion.",
    
    "Diameter of Largest Connected Component": "This is the longest shortest path between any two nodes in the largest cluster. A low diameter means that every group is easily reachable from any other."
}

# Node-Level Metrics
# Calcul des métriques de centralité
degree_dict = dict(G_groups.degree())
betweenness_dict = nx.betweenness_centrality(G_groups, weight='weight')
closeness_dict = nx.closeness_centrality(G_groups)
pagerank_dict = nx.pagerank(G_groups, weight='weight')
eigenvector_dict = nx.eigenvector_centrality(G_groups, weight='weight')

# DataFrame des métriques de centralité
centrality_full_df = pd.DataFrame({
    'Degree': degree_dict,
    'Betweenness': betweenness_dict,
    'Closeness': closeness_dict,
    'PageRank': pagerank_dict,
    'Eigenvector': eigenvector_dict
}).round(4)

# Interprétations personnalisées
centrality_stories = {
    "AGT": {
        "Degree": "AGT actively collaborates with multiple teams, ensuring its strong presence across I2M.",
        "Betweenness": "AGT plays a key role as a bridge between different parts of the institute, facilitating collaboration flows.",
        "Closeness": "AGT is at equal reach to all others — it's embedded at the heart of I2M.",
        "PageRank": "AGT is modestly connected to influential groups — its position is solid, but not dominant.",
        "Eigenvector": "AGT connects with well-connected teams, although its central influence remains moderate."
    },
    "AA": {
        "Degree": "AA has a broad reach across I2M, with active links to several other groups.",
        "Betweenness": "While not a key bridge, AA ensures redundancy in the network’s pathways.",
        "Closeness": "AA maintains perfect reachability to all others — typical of a fully connected network.",
        "PageRank": "AA ranks high in influence, sitting close to the core of I2M's collaborative ecosystem.",
        "Eigenvector": "AA shines in eigenvector centrality — it’s a major hub connected to other hubs."
    },
    "ALEA": {
        "Degree": "ALEA collaborates widely, making it one of I2M’s pillars.",
        "Betweenness": "Though not the main connector, ALEA participates in many collaboration paths.",
        "Closeness": "ALEA is equally accessible from all parts of the network — a signature of integration.",
        "PageRank": "ALEA’s strong position means it’s well-placed in both quantity and quality of links.",
        "Eigenvector": "ALEA is a power center — its influence radiates through top-level collaborations."
    },
    "GDAC": {
        "Degree": "GDAC participates actively in co-publications with other groups.",
        "Betweenness": "GDAC occasionally acts as a connector — but not the main bridge.",
        "Closeness": "GDAC, like its peers, is equally reachable — thanks to the network’s tight design.",
        "PageRank": "Its influence is steady — GDAC is well-placed in the scientific web.",
        "Eigenvector": "GDAC is connected to key nodes, giving it a balanced and stable presence."
    },
    "AGLR": {
        "Degree": "AGLR holds equal collaboration power as the rest — in a fully connected network.",
        "Betweenness": "AGLR plays a smaller role in bridging, relying more on others to connect the dots.",
        "Closeness": "Being close to all others, AGLR stays deeply embedded in the network.",
        "PageRank": "Its PageRank is moderate — AGLR is well-positioned but not at the center of gravity.",
        "Eigenvector": "AGLR benefits from being connected to influential peers — though not at the very top."
    }
}






# Community Detection

partition = community_louvain.best_partition(G_groups, weight='weight')
nx.set_node_attributes(G_groups, partition, "community")

community_df = pd.DataFrame({
    "Group": list(partition.keys()),
    "Community": list(partition.values())
}).sort_values("Community")

community_members = community_df.groupby("Community")["Group"].apply(list).to_dict()
unique_communities = sorted(community_df['Community'].unique())
colors = ['#FFB3BA', '#BAE1FF', '#BAFFC9', '#FFFFBA', '#FFDFBA']

def generate_cyto_community_elements():
    cyto_elements_comm = []
    for node in G_groups.nodes():
        community = partition[node]
        cyto_elements_comm.append({
            'data': {'id': node, 'label': node},
            'classes': f"comm-{community}"
        })
    for u, v, data in G_groups.edges(data=True):
        cyto_elements_comm.append({
            'data': {
                'source': u,
                'target': v,
                'weight': data['weight'],
                'tooltip': data['weight']
            }
        })
    return cyto_elements_comm




# Assortativity and Clustering Coefficient

assortativity = nx.degree_pearson_correlation_coefficient(G_groups)

global_clustering = nx.average_clustering(G_groups, weight='weight')

network_structure_metrics = {
    "Assortativity": {
        "value": assortativity,
        "interpretation": (
            "The assortativity coefficient indicates whether research groups with similar collaboration levels "
            "tend to collaborate. A positive value means similar groups connect together; a negative one indicates "
            "diverse mixing. In this case, the value is undefined — likely due to a uniform network where all nodes "
            "have the same degree."
        ) if pd.isna(assortativity) else (
            "Groups tend to connect with others of similar degree (positive assortativity)" if assortativity > 0 else
            "Groups tend to connect with others of different degree (negative assortativity)"
        )
    },
    "Global Clustering Coefficient": {
        "value": global_clustering,
        "interpretation": (
            f"The global clustering coefficient is {global_clustering:.4f}, which suggests a "
            f"{'low' if global_clustering < 0.3 else 'moderate' if global_clustering < 0.6 else 'high'} tendency "
            f"for groups to form interconnected collaboration triangles."
        )
    }
}

# Qualitative Roles of Research Groups
clustering_dict = nx.clustering(G_groups, weight='weight')

centrality_df = centrality_full_df.copy()
centrality_df['Clustering'] = pd.Series(clustering_dict)

def assign_role(row):
    if row['Betweenness'] > 0.3:
        return 'Bridge'
    elif row['Clustering'] > 0.5:
        return 'Clustered Group'
    elif row['Degree'] == centrality_df['Degree'].max() and row['Betweenness'] < 0.05:
        return 'Connector'
    else:
        return 'Other'

centrality_df['Role'] = centrality_df.apply(assign_role, axis=1)
centrality_df = centrality_df[['Degree', 'Betweenness', 'Clustering', 'Role']]

role_explanations = {
    'Bridge': " This group acts as a strategic bridge, connecting distant parts of the network. It likely facilitates cross-team or interdisciplinary collaboration.",
    'Connector': " This group collaborates broadly — it has many connections, but does not serve as a key bridge. It likely spreads ideas widely across I2M.",
    'Clustered Group': " This group is part of a close-knit team. It shows strong local collaboration and scientific cohesion.",
    'Other': " This group doesn’t fit the standard roles. It may have a unique collaboration pattern within I2M."
}

# Temporal Evolution of Group Collaboration


from collections import defaultdict

periods = {
    'P1 (2014–2017)': (2014, 2017),
    'P2 (2018–2020)': (2018, 2020),
    'P3 (2021–2024)': (2021, 2024)
}

temporal_networks = {}
temporal_explanations = {
    "P1 (2014–2017)": "In the early years, the collaboration network is modest and selective. One partnership stands out clearly: AA and AGLR, who co-authored 15 publications, forming the dominant alliance of this phase. Other connections, like ALEA–AGT and AA–ALEA, start to appear, but remain sporadic and low in intensity. It’s a time of emerging links — tentative, exploratory.",
    "P2 (2018–2020)": "The landscape begins to shift. Bilateral partnerships give way to broader exchanges. Strong ties form between AA and ALEA (10), ALEA and GDAC (8), and AA and GDAC (7) — suggesting a move toward more diversified and distributed collaborations. The AA–AGLR connection, once dominant, softens slightly, making space for a more balanced internal dynamic.",
    "P3 (2021–2024)": "The network reaches maturity. Collaborations grow not only stronger but structurally richer. A triangular core emerges, centered on AA, ALEA, and GDAC, with links like AA–ALEA (14), ALEA–GDAC (13), and AA–GDAC (6). Meanwhile, AGT and AGLR become more active, forging consistent but less central connections. This period reflects a truly integrated collaboration culture."
}

for label, (start_year, end_year) in periods.items():
    period_df = df[
        (df['Publication Date'] >= start_year) &
        (df['Publication Date'] <= end_year)
    ]

    period_df['Research Groups'] = period_df['Research Groups'].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else x)
    group_pairs = []

    for groups in period_df['Research Groups']:
        if isinstance(groups, list) and len(groups) > 1:
            pairs = combinations(sorted(set(groups)), 2)
            group_pairs.extend(pairs)

    pair_counts = Counter(group_pairs)

    nodes = sorted(set([g for pair in pair_counts for g in pair]))
    elements = [{'data': {'id': node, 'label': node}} for node in nodes]

    for (g1, g2), weight in pair_counts.items():
        elements.append({'data': {
            'source': g1,
            'target': g2,
            'weight': weight,
            'tooltip': weight
        }})

    temporal_networks[label] = elements

# 2.2 Collaboration Between Authors¶

# Author Collaboration Network Construction

def safe_eval(x):
    if isinstance(x, list):
        return x
    if pd.isna(x):
        return []
    try:
        return ast.literal_eval(x)
    except Exception:
        return []

df["I2M Authors"] = df["I2M Authors"].apply(safe_eval)




G_authors = nx.Graph()
for authors in df["I2M Authors"]:
    if len(authors) >= 2:
        for pair in combinations(sorted(set(authors)), 2):
            if G_authors.has_edge(*pair):
                G_authors[pair[0]][pair[1]]["weight"] += 1
            else:
                G_authors.add_edge(pair[0], pair[1], weight=1)

all_author_names = sorted(G_authors.nodes())

# Core Network Metrics

num_nodes_authors = G_authors.number_of_nodes()
num_edges_authors = G_authors.number_of_edges()
avg_degree_authors = sum(dict(G_authors.degree()).values()) / num_nodes_authors
density_authors = nx.density(G_authors)
num_components_authors = nx.number_connected_components(G_authors)

author_metrics = {
    "Number of Nodes (Authors)": f"{num_nodes_authors}",
    "Number of Edges (Co-authorships)": f"{num_edges_authors}",
    "Average Degree": f"{avg_degree_authors:.2f}",
    "Density": f"{density_authors:.4f}",
    "Number of Connected Components": f"{num_components_authors}"
}

author_metric_explanations = {
    "Number of Nodes (Authors)": "342 I2M authors are included in the network. Each has collaborated with at least one other internal I2M member.",
    "Number of Edges (Co-authorships)": "494 co-authorship links exist between these authors, indicating joint publications.",
    "Average Degree": "Each author has collaborated with about 2.89 others on average — showing moderate internal density.",
    "Density": "The network’s density is 0.0085 — very sparse, as expected in academic settings with focused collaborations.",
    "Number of Connected Components": "17 groups of authors form disconnected subgraphs — reflecting isolated teams or occasional collaboration gaps."
}

# Node-Level Centrality
degree_centrality = nx.degree_centrality(G_authors)
betweenness_centrality = nx.betweenness_centrality(G_authors, weight='weight')
closeness_centrality = nx.closeness_centrality(G_authors)
pagerank_centrality = nx.pagerank(G_authors, weight='weight')
eigenvector_centrality = nx.eigenvector_centrality(G_authors, weight='weight', max_iter=1000)

centrality_authors_df = pd.DataFrame({
    'Degree': pd.Series(degree_centrality),
    'Betweenness': pd.Series(betweenness_centrality),
    'Closeness': pd.Series(closeness_centrality),
    'PageRank': pd.Series(pagerank_centrality),
    'Eigenvector': pd.Series(eigenvector_centrality)
}).round(4)

degree_threshold = centrality_authors_df['Degree'].quantile(0.90)
eigenvector_threshold = centrality_authors_df['Eigenvector'].quantile(0.90)
betweenness_threshold = 0.01

def assign_author_role(row):
    if row['Betweenness'] > betweenness_threshold:
        return 'Bridge'
    elif row['Degree'] > degree_threshold:
        return 'Hub'
    elif row['Eigenvector'] > eigenvector_threshold:
        return 'Influencer'
    else:
        return 'Peripheral'

centrality_authors_df['Role'] = centrality_authors_df.apply(assign_author_role, axis=1)
centrality_authors_df.index.name = 'Author'
centrality_authors_df = centrality_authors_df.reset_index()

#  Identify Key Connectors
# Trier les auteurs par centralité (Betweenness) et remettre l'index à plat
top_connectors = centrality_authors_df.sort_values(by='Betweenness', ascending=False).reset_index()
top_connectors = top_connectors[['Author', 'Betweenness', 'Role']].head(15)


def generate_author_cards(df):
    cards = []
    for _, row in df.iterrows():
        cards.append(
            html.Div([
                html.H4(row["Author"], style={"marginBottom": "5px", "color": "#d35400"}),
                html.P(f"Betweenness Centrality: {row['Betweenness']:.3f}", style={"marginBottom": "3px"}),
                html.P(f"Role: {row['Role']}", style={"marginBottom": "0"})
            ], style={
                "backgroundColor": "#fdfefe",
                "padding": "15px",
                "marginBottom": "15px",
                "border": "1px solid #e0e0e0",
                "borderLeft": "6px solid #e74c3c",  
                "borderRadius": "8px",
                "boxShadow": "0px 2px 5px rgba(0,0,0,0.05)"
            })
        )
    return cards


# Community Detection among Authors


global_author_partition = community_louvain.best_partition(G_authors, weight='weight')

global_author_community_df = pd.DataFrame({
    'Author': list(global_author_partition.keys()),
    'Community': list(global_author_partition.values())
})

global_community_centrality_df = pd.merge(
    global_author_community_df, centrality_authors_df, on="Author"
)

global_community_options = sorted(global_community_centrality_df["Community"].unique())
global_community_dropdown_options = [{"label": f"Community {c}", "value": c} for c in global_community_options]

# Focused Subgraphs

from collections import defaultdict
import networkx as nx
import community as community_louvain

focused_author_partition = community_louvain.best_partition(G_authors)
nx.set_node_attributes(G_authors, focused_author_partition, 'community')

from collections import defaultdict
focused_communities = defaultdict(list)
for author, cid in focused_author_partition.items():
    focused_communities[cid].append(author)

focused_top_communities = sorted(focused_communities.items(), key=lambda x: len(x[1]), reverse=True)[:3]
focused_community_dropdown_options = [{"label": f"Community {i+1} ({len(comm[1])} authors)", "value": i}
                                      for i, comm in enumerate(focused_top_communities)]
focused_community_members_lookup = {i: sorted(comm[1]) for i, comm in enumerate(focused_top_communities)}

def detect_interpretation_community(members):
    members_set = set(a.lower() for a in members)

    if any(a in members_set for a in ["clara bourot", "laurence reboul", "jean marc freyermuth"]):
        return "This group appears to be a close-knit and productive research cell, shaped around strong collaborations. Authors like Clara Bourot, Laurence Reboul, and Jean-Marc Freyermuth anchor the group — likely tied by shared themes or long-term projects."
    elif any(a in members_set for a in ["elisabeth remy", "anais baudot", "laurent tichit"]):
        return "This community reflects a densely connected and interdisciplinary team. Figures such as Elisabeth Remy, Anaïs Baudot, and Laurent Tichit point to cross-cutting collaborations — possibly blending biology, computation, and medical science."
    elif any(a in members_set for a in ["badih ghattas", "erwan rousseau", "quentin ghibaudo"]):
        return "This subnetwork reveals a methodologically-oriented cluster. With researchers like Badih Ghattas, Erwan Rousseau, and Quentin Ghibaudo, this group may specialize in statistical modeling, data-driven methodologies, or applied mathematics."
    else:
        return "This community is structurally relevant within the I2M collaboration network. While it doesn’t feature a dominant core, its composition still suggests coordinated research activity worth deeper exploration."

cliques = list(nx.find_cliques(G_authors))
largest_cliques = [clq for clq in cliques if len(clq) >= 4]
top_cliques = sorted(largest_cliques, key=len, reverse=True)[:3]
clique_dropdown_options = [{"label": f"Clique {i+1} ({len(clq)} authors)", "value": i}
                           for i, clq in enumerate(top_cliques)]
clique_members_lookup = {i: sorted(clq) for i, clq in enumerate(top_cliques)}

clique_interpretations = {
    0: "Clique 1 showcases a tightly interwoven micro-network. With names like C. Crisci, Alexandre Boritchev, and Quentin Ghibaudo, this group suggests an agile and productive team with strong co-authorship bonds.",
    1: "Clique 2 highlights a robust scientific alliance anchored by Laurent Tichit and Anaïs Baudot. Their co-publication ties likely stem from collaborative projects with high interdisciplinarity.",
    2: "Clique 3 overlaps with one of the core Louvain communities. Its structure confirms the pivotal role of Clara Bourot and Jean-Marc Freyermuth as key contributors to I2M’s collaborative engine."
}

# Temporal Trends of Author Collaborations

import ast
import networkx as nx
import pandas as pd
import plotly.graph_objects as go

graphs_by_year = {}

for year in sorted(df['Publication Date'].dropna().unique()):
    yearly_df = df[df['Publication Date'] == year]
    G_year = nx.Graph()

    for authors in yearly_df['I2M Authors'].dropna():
        if isinstance(authors, str):
            authors_list = ast.literal_eval(authors)
        elif isinstance(authors, list):
            authors_list = authors
        else:
            continue

        for i in range(len(authors_list)):
            for j in range(i + 1, len(authors_list)):
                a1, a2 = sorted([authors_list[i], authors_list[j]])
                if G_year.has_edge(a1, a2):
                    G_year[a1][a2]['weight'] += 1
                else:
                    G_year.add_edge(a1, a2, weight=1)

    graphs_by_year[year] = G_year

years = []
num_nodes = []
num_edges = []
densities = []

for year, G in sorted(graphs_by_year.items()):
    years.append(year)
    num_nodes.append(G.number_of_nodes())
    num_edges.append(G.number_of_edges())
    densities.append(nx.density(G))

evolution_df = pd.DataFrame({
    "Year": years,
    "Number of Authors": num_nodes,
    "Number of Co-authorships": num_edges,
    "Graph Density": densities
})

evolution_df["Graph Density (%)"] = (evolution_df["Graph Density"] * 100).round(2)

initial_fig = go.Figure(
    data=[
        go.Scatter(x=evolution_df["Year"], y=[0]*len(evolution_df), mode='lines+markers',
                   name='Number of Authors', line=dict(color="#1f77b4")),
        go.Scatter(x=evolution_df["Year"], y=[0]*len(evolution_df), mode='lines+markers',
                   name='Number of Co-authorships', line=dict(color="#ff7f0e")),
        go.Scatter(x=evolution_df["Year"], y=[0]*len(evolution_df), mode='lines+markers',
                   name='Graph Density (%)', line=dict(color="#2ca02c"), yaxis='y2'),
    ],
    layout=go.Layout(
        title="Evolution of Internal Collaboration Network Over Time",
        xaxis=dict(title="Year"),
        yaxis=dict(title="Authors & Co-authorships"),
        yaxis2=dict(title="Graph Density (%)", overlaying='y', side='right'),
        plot_bgcolor='white',
        paper_bgcolor='white',
        font=dict(color="#2c3e50"),
        margin=dict(t=80, b=40, l=40, r=20)
    )
)

temporal_observations = """
What we discover is not just data — it’s a narrative of scientific life:
•  From 2014 to 2019, the network steadily expanded. More I2M authors joined the collaboration graph each year, forming new connections and nurturing scientific dialogue.
• 2022 marks the peak: the highest number of active authors and co-authorship links in the entire decade. A moment where collaboration was at its most vibrant.
•  2020 stands out as a rupture. Both the number of authors and their connections dropped sharply — a visible trace of the COVID-19 pandemic disrupting usual research flows.
• Post-2020, the network slowly begins to rebuild. Collaborations resume, yet they remain below their former levels, perhaps reflecting ongoing constraints or a shift in research dynamics.
•  Throughout all years, network density stays low — a natural trait of co-authorship networks, where most researchers collaborate with just a handful of peers rather than with everyone.

This temporal evolution offers a unique lens into the rhythms of collaboration at I2M: a story of growth, shock, and gradual recovery — told not through words, but through the structure of a network.
"""



# Identification of External Co-Authors
all_external_names = df.explode('External Authors')['External Authors'].dropna().str.lower().str.strip()
unique_external_names = all_external_names.unique()

all_i2m_names = df.explode('I2M Authors')['I2M Authors'].dropna().str.lower().str.strip()
unique_i2m_names = all_i2m_names.unique()

external_stats = {
    "Unique External Authors": {
        "value": len(unique_external_names),
        "interpretation": (
            f"We identified a total of {len(unique_external_names):,} unique external co-authors. "
            "This striking figure shows that for every I2M researcher, there are multiple collaborators outside the institute, spanning different institutions, countries, and domains. "
            "Rather than working in isolation, I2M operates like a collaborative hub, engaging with the wider scientific ecosystem. "
            "The size of this external author pool illustrates the strategic importance of openness, cross-disciplinary exchange, and global partnerships in the lab’s research output."
        )
    },
    "Unique I2M Authors": {
        "value": len(unique_i2m_names),
        "interpretation": (
            f"There are {len(unique_i2m_names)} unique I2M-affiliated authors in the dataset. "
            "These are the researchers who form the intellectual backbone of the institute — the internal force driving innovation, sharing expertise, and building long-term academic continuity."
        )
    }
}

external_dropdown_options = [
    {"label": "Unique External Authors", "value": "Unique External Authors"},
    {"label": "Unique I2M Authors", "value": "Unique I2M Authors"}
]


# Identify Top External Collaborators
df['External Authors'] = df['External Authors'].apply(
    lambda x: ast.literal_eval(x) if isinstance(x, str) and x.startswith('[') else x
)

external_authors_exploded = df.explode('External Authors')[['External Authors', 'Affiliations']].dropna()
external_authors_exploded['External Authors'] = external_authors_exploded['External Authors'].str.lower().str.strip()

def parse_affiliations(aff):
    try:
        parsed = ast.literal_eval(aff)
        if isinstance(parsed, list):
            return parsed
        elif isinstance(parsed, str):
            return [parsed.strip()]
    except:
        return [str(aff).strip()]

external_authors_exploded['Affiliations'] = external_authors_exploded['Affiliations'].apply(parse_affiliations)
exploded_affiliations = external_authors_exploded.explode('Affiliations')
exploded_affiliations['Affiliations'] = exploded_affiliations['Affiliations'].str.strip()

top_affiliations = (
    exploded_affiliations
    .groupby('External Authors')['Affiliations']
    .agg(lambda x: x.mode().iloc[0] if not x.mode().empty else 'Unknown')
    .reset_index()
    .rename(columns={'External Authors': 'Author', 'Affiliations': 'Affiliation'})
)

external_collaborators = df.explode('External Authors')['External Authors'].dropna().str.lower().str.strip()
top_external_collaborators = external_collaborators.value_counts().head(20).reset_index()
top_external_collaborators.columns = ['Author', 'Publications']

top_external_collaborators = pd.merge(top_external_collaborators, top_affiliations, on='Author', how='left')

def generate_external_cards(df):
    cards = []
    for _, row in df.iterrows():
        cards.append(
            html.Div([
                html.H4(row["Author"].title(), style={"marginBottom": "5px", "color": "#117864"}),
                html.P(f"Affiliation: {row['Affiliation']}", style={"marginBottom": "6px", "fontSize": "14px"}),
                html.P(f"Number of co-authored publications: {row['Publications']}", style={"margin": 0})
            ], style={
                "backgroundColor": "#fdfefe",
                "padding": "15px",
                "marginBottom": "15px",
                "marginLeft": "auto",          
                "marginRight": "auto",         
                "maxWidth": "600px",
                "border": "1px solid #e0e0e0",
                "borderLeft": "6px solid #16a085", 
                "borderRadius": "10px",
                "boxShadow": "0px 2px 5px rgba(0,0,0,0.05)",
                "width": "100%"
            })
        )
    return cards


# Institutions by Co-publication Frequency

df['Affiliations'] = df['Affiliations'].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) and x.startswith('[') else x)
affiliations_exploded = df.explode('Affiliations')['Affiliations'].dropna().astype(str).str.lower().str.strip()

affiliations_cleaned = (
    affiliations_exploded
    .str.replace(r"^\[\]$", "", regex=True)
    .str.replace("\"", "")
    .str.replace("'", "")
    .str.strip()
)

to_exclude = ['', 'unknown']
affiliations_cleaned = affiliations_cleaned[~affiliations_cleaned.isin(to_exclude)]

top_affiliations_cleaned = affiliations_cleaned.value_counts().head(30).reset_index()
top_affiliations_cleaned.columns = ['Institution', 'Frequency']

def generate_affiliation_cards(df):
    cards = []
    for _, row in df.iterrows():
        cards.append(
            html.Div([
                html.H4(row["Institution"].title(), style={"marginBottom": "5px", "color": "#f39c12"}),
                html.P(f"Number of co-authored publications: {row['Frequency']}", style={"margin": 0})
            ], style={
                "backgroundColor": "#fdfefe",
                "padding": "15px",
                "marginBottom": "15px",
                "marginLeft": "auto",          
                "marginRight": "auto",         
                "maxWidth": "600px",
                "border": "1px solid #e0e0e0",
                "borderLeft": "6px solid #f39c12", 
                "borderRadius": "10px",
                "boxShadow": "0px 2px 5px rgba(0,0,0,0.05)",
                "width": "100%"
            })
        )
    return cards

# Visualization of External Collaboration Network
def safe_eval(x):
    if isinstance(x, list):
        return x
    if pd.isna(x):
        return []
    try:
        return ast.literal_eval(x)
    except Exception:
        return []

df['I2M Authors'] = df['I2M Authors'].apply(safe_eval)
df['External Authors'] = df['External Authors'].apply(safe_eval)

G_external = nx.Graph()

for _, row in df.iterrows():
    i2m_authors = list(set(row['I2M Authors'] or []))
    external_authors = list(set(row['External Authors'] or []))
    for i2m in i2m_authors:
        for ext in external_authors:
            if G_external.has_edge(i2m, ext):
                G_external[i2m][ext]['weight'] += 1
            else:
                G_external.add_edge(i2m, ext, weight=1)

external_counts = {}
for u, v, d in G_external.edges(data=True):
    ext_node = v if v not in df['I2M Authors'].explode().unique() else u
    external_counts[ext_node] = external_counts.get(ext_node, 0) + 1

all_external_authors = sorted(external_counts.keys())
network_dropdown_options = [{"label": name, "value": name} for name in all_external_authors]




# Focus on Strategic Collaborations (UMRs at AMU)
affiliations_cleaned = df.explode('Affiliations')['Affiliations'].dropna().str.lower().str.strip()
affiliations_cleaned = affiliations_cleaned.str.replace(r'[\[\]\'\"]', '', regex=True)
affiliations_cleaned = affiliations_cleaned.str.replace(r'\s+', ' ', regex=True).str.strip()

amu_keywords = [
    'aix-marseille', 'aix marseille', 'université d\'aix-marseille',
    'aix-marseille université', 'univ. aix-marseille', 'univ aix marseille',
    'univ d\'aix marseille', 'université d’aix marseille', 'u. aix-marseille'
]


amu_affiliations = affiliations_cleaned[affiliations_cleaned.str.contains('|'.join(amu_keywords))]

amu_top_counts = amu_affiliations.value_counts().head(15)
amu_labels = amu_top_counts.index.to_series()

# Keyword Extraction from Titles


nltk.download('punkt')
nltk.download('stopwords')

tokenizer = TreebankWordTokenizer()


titles = df['Title'].dropna().astype(str).str.lower()

custom_stopwords = set(stopwords.words('english')) | {
    'non', 'des', 'sur', 'les', 'pour', 'dans', 'avec', 'nous', 'dont', 'par', 'en', 'une', 'ce', 'cette', 'est', 'qui'
}

word_counts = Counter()

for title in titles:
    tokens = tokenizer.tokenize(re.sub(r'[^a-zA-Z\s]', '', title))
    filtered = [w for w in tokens if w not in custom_stopwords and len(w) > 2]
    word_counts.update(filtered)


def generate_wordcloud_base64(top_k=100):
    wordcloud = WordCloud(width=1000, height=500, background_color='white',
                          max_words=top_k).generate_from_frequencies(word_counts)
    buffer = BytesIO()
    wordcloud.to_image().save(buffer, format='PNG')
    return base64.b64encode(buffer.getvalue()).decode()

#  Dominant Themes by Research Group


custom_stopwords = set(stopwords.words('english')) | {
    'non', 'des', 'sur', 'les', 'pour', 'dans', 'avec', 'nous', 'dont', 'par',
    'en', 'une', 'ce', 'cette', 'est', 'qui', 'et', 'un'
}

group_keywords = defaultdict(Counter)
df_grouped = df[['Title', 'Research Groups']].dropna()

for _, row in df_grouped.iterrows():
    title = str(row['Title']).lower()
    raw_groups = row['Research Groups']

    if isinstance(raw_groups, list):
        groups = [str(g).strip() for g in raw_groups]
    elif isinstance(raw_groups, str):
        groups = [g.strip() for g in re.split(r';|,|/', raw_groups)]
    else:
        continue

    tokens = tokenizer.tokenize(re.sub(r'[^a-zA-Z\s]', '', title))  # ✅ la bonne version
    filtered = [w for w in tokens if w not in custom_stopwords and len(w) > 2]

    for group in groups:
        group_keywords[group].update(filtered)

group_wordclouds = {}
for group, keywords in group_keywords.items():
    if len(keywords) < 5:
        continue
    wc = WordCloud(width=800, height=400, background_color='white').generate_from_frequencies(keywords)
    buffer = BytesIO()
    wc.to_image().save(buffer, format='PNG')
    encoded_img = base64.b64encode(buffer.getvalue()).decode()
    group_wordclouds[group] = encoded_img

group_texts = {
    "AGT": html.Div([
        html.P([
            html.Span("Frequently appearing terms: ", style={"fontWeight": "bold"}),
            html.Span("spaces, surfaces, complex, bundles, manifolds, geometry")
        ]),
        html.P("This group delves into the abstract beauty of geometric structures — studying topological spaces, complex manifolds, and fiber bundles. "
               "Their research weaves together geometry and topology with elegant mathematical rigor.")
    ]),
    "ALEA": html.Div([
        html.P([
            html.Span("Frequently appearing terms: ", style={"fontWeight": "bold"}),
            html.Span("random, model, adaptive, application, walks, process")
        ]),
        html.P("ALEA's scientific landscape is shaped by randomness. From stochastic processes to probabilistic algorithms and adaptive models, "
               "this group explores uncertainty with deep mathematical insight and real-world relevance.")
    ]),
    "AA": html.Div([
        html.P([
            html.Span("Frequently appearing terms: ", style={"fontWeight": "bold"}),
            html.Span("equations, model, compressible, flows, scheme")
        ]),
        html.P("At the heart of AA's work lies the motion of fluids and the language of partial differential equations. "
               "Their research tackles compressible flows, mathematical modeling, and the development of robust numerical schemes for complex physical systems.")
    ]),
    "AGLR": html.Div([
        html.P([
            html.Span("Frequently appearing terms: ", style={"fontWeight": "bold"}),
            html.Span("codes, fields, finite, cyclic, curves")
        ]),
        html.P("AGLR investigates the foundations of information and number theory. Through topics such as finite fields, algebraic curves, and coding theory, "
               "this group unravels the mathematical structures behind secure communication and arithmetic geometry.")
    ]),
    "GDAC": html.Div([
        html.P([
            html.Span("Frequently appearing terms: ", style={"fontWeight": "bold"}),
            html.Span("groups,measures, automata, dynamics, complexity, infinite")
        ]),
        html.P("GDAC navigates the intersections of algebra, logic, and dynamics. With a focus on group theory, automata, and infinite structures, "
               "this group explores how complexity arises from simple mathematical rules and discrete systems.")
    ])
}

# Evolution of Scientific Topics Over Time

custom_stopwords = set(stopwords.words('english')) | {
    'non', 'des', 'sur', 'les', 'pour', 'dans', 'avec', 'nous', 'dont', 'par',
    'en', 'une', 'ce', 'cette', 'est', 'qui', 'et', 'un'
}

year_keywords = defaultdict(Counter)
df_year_title = df[['Title', 'Publication Date']].dropna()

for _, row in df_year_title.iterrows():
    title = str(row['Title']).lower()
    year = int(row['Publication Date'])

    tokens = tokenizer.tokenize(re.sub(r'[^a-zA-Z\s]', '', title))  # ✅ la bonne version
    filtered = [w for w in tokens if w not in custom_stopwords and len(w) > 2]
    year_keywords[year].update(filtered)

year_wordclouds = {}
for year, keywords in sorted(year_keywords.items()):
    if len(keywords) < 3:
        continue
    wordcloud = WordCloud(width=1000, height=500, background_color='white',
                          max_words=100).generate_from_frequencies(keywords)
    buffer = BytesIO()
    wordcloud.to_image().save(buffer, format='PNG')
    encoded_img = base64.b64encode(buffer.getvalue()).decode()
    year_wordclouds[year] = encoded_img

















######################################################################################################################

app = dash.Dash(__name__)
app.title = "Bibliometric Dashboard"

app.layout = html.Div(
    style={
        "backgroundColor": "#eef2f7",
        "padding": "40px",
        "fontFamily": "Segoe UI, sans-serif"
    },
    children=[
        # Titre principal
        html.H1(
            "Bibliometric Dashboard of I2M Publications",
            style={
                "textAlign": "center",
                "color": "#34495e",
                "marginBottom": "10px",
                "fontSize": "36px",
                "letterSpacing": "1px"
            }
        ),

        # Sous-titre
        html.H4(
            "By Sabine Mansour",
            style={
                "textAlign": "center",
                "color": "#34495e",
                "marginBottom": "40px",
                "fontStyle": "bold"
            }
        ),

        # Introduction 
        html.Div(
    children=[
        html.P("Welcome to the I2M Bibliometric Dashboard — a journey through a decade of scientific discovery."),
        html.P("This interactive platform invites you to explore the research landscape shaped by the I2M laboratory. From publication trends and document types to collaborative patterns both internal and external, each chart reveals part of the institute’s scientific identity."),
        html.P("Whether you're curious about the most prolific research groups, key authors, or the balance between internal and global partnerships, this dashboard provides the tools to dive deeper. Use the filters, dropdowns, and visualizations to uncover insights and follow the evolving story of I2M’s contributions to science."),
    ],
    style={
        "maxWidth": "900px",
        "margin": "auto",
        "fontSize": "17px",
        "lineHeight": "1.8",
        "color": "#2c3e50",
        "backgroundColor": "#ffffff",
        "padding": "30px",
        "borderRadius": "15px",
        "boxShadow": "0px 5px 20px rgba(0, 0, 0, 0.1)",
        "borderLeft": "6px solid #16a085",
        "marginBottom": "70px"
    }
),
        #I2M
html.Div(
    style={
        "backgroundColor": "#ffffff",
        "padding": "30px",
        "marginTop": "50px",
        "marginBottom": "40px",
        "borderLeft": "6px solid #e74c3c",
        "boxShadow": "0px 5px 20px rgba(0,0,0,0.05)",
        "borderRadius": "10px",
        "maxWidth": "950px",
        "margin": "auto"
    },
    children=[
        html.H2(" About I2M: The Institute Behind the Science", style={
            "color": "#2c3e50", "fontSize": "28px", "marginBottom": "15px"}),

        html.P("Before we dive into charts and metrics, let's take a moment to understand the structure of the I2M itself."),
        html.P("The Institut de Mathématiques de Marseille (I2M) is a joint research unit (UMR 7373) affiliated with CNRS, Aix-Marseille University, and Centrale Marseille. "
               "It gathers over 130 faculty members, 30 CNRS researchers, and around 80 doctoral and postdoctoral scholars."),
        html.P("Organized into five scientific groups, I2M’s research spans from pure theory to applied problems, creating a fertile ground for interdisciplinary collaborations."),

        dcc.Dropdown(
            id="group-dropdown",
            options=[{"label": name, "value": name} for name in group_descriptions.keys()],
            placeholder="Select a research group...",
            style={"marginTop": "10px"}
        ),

        html.Div(
            id="group-description-box",
            style={
                "marginTop": "25px",
                "border": "2px dashed #e74c3c",
                "padding": "20px",
                "borderRadius": "8px",
                "backgroundColor": "#ffffff",
                "color": "#2c3e50",
                "fontSize": "16px",
                "lineHeight": "1.6"
            }
        )
    ]
),


       html.Div(style={"height": "80px"}),  # espace vertical de 80px




        # 1. Overview of I2M Scientific Output
        html.Div(
    style={
        "backgroundColor": "#ffffff",
        "padding": "30px",
        "marginTop": "50px",
        "marginBottom": "40px",
        "borderLeft": "6px solid #f39c12",
        "boxShadow": "0px 5px 20px rgba(0,0,0,0.05)",
        "borderRadius": "10px",
        "maxWidth": "950px",
        "margin": "auto"
    },
    children=[
        html.H2("1.Tracing the Footprint: An Overview of I2M’s Scientific Output",
                style={"color": "#2c3e50", "fontSize": "28px", "marginBottom": "15px"}),

        html.P("Every story begins with context. Before unraveling the intricate web of collaborations, we step back to take in the broader picture — the scientific footprint of the I2M."),
        html.P("How prolific has the institute been over the years? What were its golden eras of productivity? Who are the groups and individuals who have shaped its scientific journey?"),
        html.P("This first chapter sets the stage. By exploring the volume, timing, and key contributors to I2M's publications, we begin to uncover the patterns that will guide us through the deeper layers of this bibliometric exploration."),
    ]
),

        # Publications per Year
        html.Div(
    style={
        "backgroundColor": "#ffffff",
        "padding": "30px",
        "marginTop": "60px",
        "marginBottom": "20px",
        "borderLeft": "6px solid #16a085",
        "boxShadow": "0px 2px 10px rgba(0,0,0,0.05)",
        "borderRadius": "10px",
        "maxWidth": "920px",
        "margin": "auto"
    },
    children=[
        html.H3("A Decade of Discovery: Tracking I2M’s Scientific Pulse", style={"color": "#2c3e50", "marginBottom": "20px"}),

        html.P("Over the past ten years, I2M’s scientific activity has drawn its rhythm year by year — sometimes steady, sometimes surging. In this section, we take a step back to observe the evolution of that rhythm. How has the institute’s research production changed over time? Which years marked a scientific high point? By tracing this timeline, we begin to sense the heartbeat of I2M’s scientific journey, identifying moments of acceleration and periods of pause. This temporal lens is our starting point — revealing not just numbers, but momentum.",
               style={"fontSize": "16px", "lineHeight": "1.8", "color": "#2c3e50"}),

        html.Button("Animate Chart", id="animate-button", n_clicks=0, style={
            "marginBottom": "20px",
            "backgroundColor": "#16a085",
            "color": "white",
            "border": "none",
            "padding": "10px 20px",
            "fontSize": "16px",
            "borderRadius": "5px",
            "cursor": "pointer"
        }),

        dcc.Graph(id='pubs-bar-graph', figure=bar_fig),

        html.P("Looking at the decade-long timeline of I2M’s publications, a story of resilience and steady dedication unfolds. The year 2017 stands out as a landmark, with the highest number of publications — a clear peak in scientific activity. Years like 2016, 2018, and 2021 also shine, reflecting strong momentum. Then comes 2020 — a sharp dip in output, likely echoing the global disruption brought by the COVID-19 pandemic. Yet, what follows is telling: a quick rebound in 2021, and a return to stability in the years that followed. This pattern not only reflects numbers, but the capacity of the I2M community to adapt, persevere, and keep contributing to science even through uncertain times.",
               style={"fontSize": "16px", "lineHeight": "1.8", "color": "#2c3e50", "marginTop": "25px"}),
    ]
),

        # Document Type Distribution
        html.Div(
    style={
        "backgroundColor": "#ffffff",
        "padding": "30px",
        "marginTop": "60px",
        "marginBottom": "20px",
        "borderLeft": "6px solid #8e44ad",
        "boxShadow": "0px 2px 10px rgba(0,0,0,0.05)",
        "borderRadius": "10px",
        "maxWidth": "920px",
        "margin": "auto"
    },
    children=[
        html.H3("Beyond the Numbers: Understanding How I2M Shares Its Science",
                style={"color": "#2c3e50", "marginBottom": "20px"}),

        html.P("Every scientific contribution takes a form — a voice, a format, a choice. In this section, we explore how I2M researchers choose to communicate their work: through peer-reviewed journal articles, conference papers, book chapters, and more. By understanding these preferred formats, we gain insight into I2M’s communication strategies — what audiences are being targeted, what types of results are being shared, and how the institute balances formal publication with broader dissemination.",
               style={"fontSize": "16px", "lineHeight": "1.8", "color": "#2c3e50"}),

        html.Button(
    "Animate Chart",
    id="animate-doc-button",
    n_clicks=0,
    style={
        "marginBottom": "20px",
        "backgroundColor": "#8e44ad",
        "color": "white",
        "border": "none",
        "padding": "10px 20px",
        "fontSize": "16px",
        "borderRadius": "5px",
        "cursor": "pointer"
    }
),
        dcc.Graph(id='doc-bar-graph', figure=doc_bar_fig),


        html.P("The data reveals a clear favorite: journal articles (ART) dominate the landscape, representing more than 65% of all outputs — a strong signal of I2M’s commitment to rigorous, peer-reviewed science. Conference papers (COMM) follow, at 8%, reflecting the value of sharing results in dynamic, real-time settings. Book chapters (COUV) also play a role, especially in interdisciplinary or long-term collaborative efforts. But not everything is neatly labeled. A significant 17% of the entries are categorized as Unknown — a reminder of the imperfections in metadata collection. These unlabeled entries may hide valuable insights, and point to future steps for improving data quality through manual review or enhanced curation.",
               style={"fontSize": "16px", "lineHeight": "1.8", "color": "#2c3e50", "marginTop": "25px"}),
    ]
),
        # Publications by Research Group
        html.Div(
    style={
        "backgroundColor": "#ffffff",
        "padding": "30px",
        "marginTop": "60px",
        "marginBottom": "20px",
        "borderLeft": "6px solid #3498db",  # violet foncé
        "boxShadow": "0px 2px 10px rgba(0,0,0,0.05)",
        "borderRadius": "10px",
        "maxWidth": "920px",
        "margin": "auto"
    },
    children=[
        html.H3("The Research Engines: Spotlight on I2M's Most Productive Teams",
                style={"color": "#2c3e50", "marginBottom": "20px"}),

        html.P("Scientific production doesn’t just happen — it’s driven by research groups working at the frontier of knowledge. Each team contributes its own rhythm, focus, and intensity to the collective output of I2M. Here, we shift our lens to these driving forces. By analyzing the number of publications by group, we uncover the internal dynamics of I2M: where the output is concentrated, which teams stand out, and how diverse the institute’s scientific engine truly is.",
               style={"fontSize": "16px", "lineHeight": "1.8", "color": "#2c3e50"}),

        html.Button(
    "Animate Chart",
    id="animate-group-button",
    n_clicks=0,
    style={
        "marginBottom": "20px",
        "backgroundColor": "#3498db",
        "color": "white",
        "border": "none",
        "padding": "10px 20px",
        "fontSize": "16px",
        "borderRadius": "5px",
        "cursor": "pointer"
    }
),
dcc.Graph(id='group-bar-graph', figure=group_bar_fig),


        html.P("The figures tell a clear story: ALEA, GDAC, and AA emerge as the institute’s most prolific research groups, each with over 600 publications. Their sustained productivity highlights both their internal strength and their central role in shaping I2M’s scientific voice. Close behind, AGLR and AGT also make strong contributions, confirming a broad base of activity across the institute. These differences likely reflect a mix of factors — from the size of the teams and the funding they receive to the nature of their scientific domains and their collaborative dynamics. To go beyond the numbers, future exploration could examine how these groups collaborate, how often their work is cited, or how their influence spreads through co-authorship networks — offering a richer understanding of I2M’s internal research landscape.",
               style={"fontSize": "16px", "lineHeight": "1.8", "color": "#2c3e50", "marginTop": "25px"}),
    ]
),
        # Publications by Author
        html.Div(
    style={
        "backgroundColor": "#ffffff",
        "padding": "30px",
        "marginTop": "60px",
        "marginBottom": "20px",
        "borderLeft": "6px solid #e67e22",  
        "boxShadow": "0px 2px 10px rgba(0,0,0,0.05)",
        "borderRadius": "10px",
        "maxWidth": "920px",
        "margin": "auto"
    },
    children=[
        html.H3("Faces Behind the Figures: Most Prolific I2M Authors",
                style={"color": "#2c3e50", "marginBottom": "20px"}),

        html.P("Behind every paper is a name, and behind every name — a researcher’s journey. To better understand the individuals who shape the scientific voice of I2M, we look at the top contributors in terms of publication count. Since many publications include multiple co-authors, we carefully parsed the data to credit each individual fairly. The result is a ranking that reflects not only productivity, but long-term engagement and scientific influence within the institute.",
               style={"fontSize": "16px", "lineHeight": "1.8", "color": "#2c3e50"}),

html.Button(
    "Animate Chart",
    id="animate-author-button",
    n_clicks=0,
    style={
        "marginBottom": "20px",
        "backgroundColor": "#e67e22",
        "color": "white",
        "border": "none",
        "padding": "10px 20px",
        "fontSize": "16px",
        "borderRadius": "5px",
        "cursor": "pointer"
    }
),
dcc.Graph(id='author-bar-graph', figure=author_fig),

        html.P("Leading the way is Patrick Solé, with an impressive 92 publications to his name — a figure that speaks to years of sustained scholarly activity. He’s followed by Kai Schneider (79), Jean-Marc Hérard (53), Étienne Pardoux (51), and François Hamel (50), all of whom have left a significant mark on I2M’s scientific output. This distribution offers more than just numbers. It reflects the strength of both established leaders and mid-career researchers working side by side. The variety of profiles in this top 20 list suggests a collaborative and dynamic research environment — one where experience and emerging voices combine to drive innovation forward.",
               style={"fontSize": "16px", "lineHeight": "1.8", "color": "#2c3e50", "marginTop": "25px"}),
    ]
),
        # Internal vs External Collaborations
       html.Div(
    style={
        "backgroundColor": "#ffffff",
        "padding": "30px",
        "marginTop": "60px",
        "marginBottom": "20px",
        "borderLeft": "6px solid #e74c3c",  # turquoise
        "boxShadow": "0px 2px 10px rgba(0,0,0,0.05)",
        "borderRadius": "10px",
        "maxWidth": "920px",
        "margin": "auto"
    },
    children=[
        html.H3("Building Bridges: A Look into I2M’s Internal and External Collaborations",
                style={"color": "#2c3e50", "marginBottom": "20px"}),

        html.P("Science thrives on connection — between people, disciplines, and institutions. To understand how I2M researchers work together, we take a closer look at the nature of their collaborations: are they working mostly within the institute, reaching outward, or blending both? This breakdown of collaboration types helps reveal the ecosystem behind the publications — not just who’s writing, but who’s working together to advance research.",
               style={"fontSize": "16px", "lineHeight": "1.8", "color": "#2c3e50"}),

        dcc.Graph(id='animated-collab-pie', figure=collab_fig),

        html.P("The donut chart reveals a clear tendency: over half of I2M’s publications (54.4%) are built on external-only collaborations. This suggests strong outward connectivity — a willingness to cross boundaries and build scientific bridges...",
               style={"fontSize": "16px", "marginTop": "20px", "color": "#2c3e50"}),

        html.P("Altogether, these results paint a picture of a dynamic and open research culture — one that values internal strength while embracing collaboration beyond its own walls.",
               style={"fontSize": "16px", "lineHeight": "1.8", "color": "#2c3e50", "marginTop": "25px"}),
    ]
),

        # Collaboration Intensity: Internal vs External
        html.Div(
    style={
        "backgroundColor": "#ffffff",
        "padding": "30px",
        "marginTop": "60px",
        "marginBottom": "20px",
        "borderLeft": "6px solid #16a085",  
        "boxShadow": "0px 2px 10px rgba(0,0,0,0.05)",
        "borderRadius": "10px",
        "maxWidth": "920px",
        "margin": "auto"
    },
    children=[
        html.H3("Collaboration at the Core: How I2M Blends Internal and External Partnerships",
                style={"color": "#2c3e50", "marginBottom": "20px"}),

        html.P("Not all collaborations are created equal — some remain within the walls of the institute, while others stretch far beyond. In this section, we explore the depth of collaborative efforts by examining how often I2M publications are the product of internal teams, external partnerships, or a blend of both. By measuring the mix of authorship in each publication, we gain insights into how I2M researchers structure their scientific relationships — whether they lean inward, reach outward, or thrive at the intersection of both.",
               style={"fontSize": "16px", "lineHeight": "1.8", "color": "#2c3e50"}),

html.Button(
    "Animate Chart",
    id="animate-collab-button",
    n_clicks=0,
    style={
        "marginBottom": "20px",
        "backgroundColor": "#16a085",
        "color": "white",
        "border": "none",
        "padding": "10px 20px",
        "fontSize": "16px",
        "borderRadius": "5px",
        "cursor": "pointer"
    }
),
dcc.Graph(id='collab-bar-graph', figure=collab_bar_fig),

        html.P("The chart tells a compelling story: most I2M publications are born from mixed collaborations — involving both internal researchers and external partners. With 1,861 publications falling into this category, it’s clear that I2M’s scientific impact is amplified by openness and exchange. In contrast, 800 publications are the product of purely internal collaboration — still a substantial number, showing that internal synergies remain a strong pillar of the institute’s research activity. As expected, almost no publications rely exclusively on external authors, given the nature of the dataset. But the dominance of mixed teams reveals something deeper: I2M doesn’t just collaborate — it integrates, connecting its internal strengths with broader scientific communities to produce impactful, collective work.",
               style={"fontSize": "16px", "lineHeight": "1.8", "color": "#2c3e50", "marginTop": "25px"}),
    ]
),
        # Conclusion de 1. Overview of I2M Scientific Output
        html.Div(
    style={
        "backgroundColor": "#ffffff",
        "padding": "30px",
        "marginTop": "60px",
        "marginBottom": "20px",
        "borderLeft": "6px solid #f39c12",  
        "boxShadow": "0px 2px 10px rgba(0,0,0,0.05)",
        "borderRadius": "10px",
        "maxWidth": "920px",
        "margin": "auto",
        "marginBottom": "70px"
    },
    children=[
        html.H2("A Decade of Scientific Storytelling at I2M",
                style={"color": "#2c3e50", "marginBottom": "20px"}),

        html.P("Over the past ten years, the I2M Institute has quietly but steadily woven a rich scientific narrative — one built on dedication, collaboration, and curiosity. As we opened the first pages of this story through our exploratory analysis, we discovered a body of work made up of more than 2,600 publications — a testament to the institute’s continuous commitment to research.",
               style={"fontSize": "16px", "lineHeight": "1.8", "color": "#2c3e50"}),

        html.P("Each year brought its own rhythm: a steady flow of publications, occasional peaks of productivity, and a noticeable pause around 2020 — a likely echo of the global standstill brought by the COVID-19 pandemic. Journal articles emerged as the institute’s favorite medium, reflecting a strong presence in peer-reviewed science and a desire to contribute to lasting knowledge.",
               style={"fontSize": "16px", "lineHeight": "1.8", "color": "#2c3e50"}),

        html.P("Within the I2M community, some research groups stood out as particularly prolific — ALEA, GDAC, AA — like recurring protagonists in this collective tale. And among the authors, we found key figures whose names appeared again and again, shaping the scientific voice of the institute.",
               style={"fontSize": "16px", "lineHeight": "1.8", "color": "#2c3e50"}),

        html.P("But this isn’t a story of isolation. Most of these publications were co-written with researchers beyond the walls of I2M, demonstrating the institute’s openness and strong connections to the broader scientific world. Interestingly, the most frequent collaborations were those that combined both internal and external partners — suggesting that I2M thrives not only on its own strengths but also on the bridges it builds.",
               style={"fontSize": "16px", "lineHeight": "1.8", "color": "#2c3e50"}),

        html.P("What emerges is a portrait of an institute grounded in solid internal expertise, yet deeply embedded in a global network of ideas and innovation.",
               style={"fontSize": "16px", "lineHeight": "1.8", "color": "#2c3e50", "marginBottom": "30px"}),

    ]
),
        # ESPACE VISUEL ENTRE LES PARTIES
html.Div(style={"height": "80px"}),  # espace vertical de 80px

# 2. Internal Collaboration Network Analysis
html.Div(
    style={
        "backgroundColor": "#ffffff",
        "padding": "30px",
        "marginTop": "50px",
        "marginBottom": "40px",
        "borderLeft": "6px solid #8e44ad",  
        "boxShadow": "0px 5px 20px rgba(0,0,0,0.05)",
        "borderRadius": "12px",
        "maxWidth": "950px",
        "margin": "auto"
    },
    children=[
        html.H2("2. Internal Collaboration Network Analysis",
                style={"color": "#2c3e50", "marginBottom": "20px"}),

        html.P("As we turn to the next chapter, our focus shifts inward. What happens within I2M? How do its teams interact, collaborate, and evolve together? Through the lens of co-authorship networks, we will explore the fabric of internal collaboration — mapping the relationships that connect research groups and individuals, identifying the central figures in this ecosystem, and uncovering hidden communities. In short, this stage of this study will let us read between the lines — to understand not just what I2M produces, but how its people work together to bring knowledge to life.",
               style={"fontSize": "16px", "lineHeight": "1.8", "color": "#2c3e50"})
    ]
),

# 2.1 Collaboration Between Research Groups
html.Div(
    style={
        "backgroundColor": "#ffffff",
        "padding": "30px",
        "marginTop": "60px",
        "marginBottom": "20px",
        "borderLeft": "6px solid #3498db",
        "boxShadow": "0px 2px 10px rgba(0,0,0,0.04)",
        "borderRadius": "10px",
        "maxWidth": "920px",
        "margin": "auto"
    },
    children=[
        html.H3("2.1 Collaboration Between Research Groups: Who Works With Whom at I2M?",
                style={"color": "#2c3e50", "marginBottom": "10px"}),

        html.P("Scientific progress rarely happens in isolation. At I2M, research groups often join forces, combining expertise to tackle complex questions. In this section, we explore these internal partnerships — examining how frequently different groups collaborate, and how these connections shape the institute’s scientific landscape.",
               style={"fontSize": "15px", "lineHeight": "1.8", "color": "#2c3e50"})
    ]
),

# Co-publication Matrix (Group to Group)
html.Div(
    style={
        "backgroundColor": "#ffffff",
        "padding": "30px",
        "marginTop": "40px",
        "marginBottom": "40px",
        "borderLeft": "6px solid #e67e22",
        "boxShadow": "0px 2px 8px rgba(0,0,0,0.03)",
        "borderRadius": "8px",
        "maxWidth": "900px",
        "margin": "auto"
    },
    children=[
        html.H3("Interactive Co-publication Explorer: Compare Research Group Collaborations",
                style={"color": "#2c3e50", "marginBottom": "20px"}),

        html.P("Use the selector below to explore how many co-authored publications exist between any two I2M research groups. "
               "This interactive view allows you to discover which teams are most closely connected through shared scientific output.",
               style={"fontSize": "15px", "lineHeight": "1.8", "color": "#2c3e50"}),

        html.Div([
            html.Div([
                html.Label("Group 1:", style={"display": "block", "marginBottom": "5px"}),
                dcc.Dropdown(
                    id='group1-dropdown',
                    options=[{"label": grp, "value": grp} for grp in unique_groups],
                    placeholder="Select Group 1",
                    style={"width": "100%"}
                )
            ], style={"display": "inline-block", "width": "45%", "marginRight": "5%"}),

            html.Div([
                html.Label("Group 2:", style={"display": "block", "marginBottom": "5px"}),
                dcc.Dropdown(
                    id='group2-dropdown',
                    options=[{"label": grp, "value": grp} for grp in unique_groups],
                    placeholder="Select Group 2",
                    style={"width": "100%"}
                )
            ], style={"display": "inline-block", "width": "45%"})
        ]),

        html.Div(
    id="co-pub-result",
    style={
        "backgroundColor": "#fdfefe",
        "padding": "25px",
        "fontSize": "18px",
        "color": "#1a5276",
        "borderRadius": "10px",
        "border": "2px dashed #e67e22",
        "textAlign": "center",
        "boxShadow": "0px 2px 8px rgba(0,0,0,0.03)",
        "maxWidth": "600px",
        "margin": "20px auto 0 auto"
    }
),




        html.P(
            "Some collaborations stand out: AA and ALEA lead with 31 co-publications, followed closely by AA and AGLR (27), and ALEA and GDAC (24). "
            "These strong connections form the backbone of I2M's internal network — a web of partnerships that we'll soon visualize through graph analysis.",
            style={"fontSize": "15px", "lineHeight": "1.8", "color": "#2c3e50", "marginTop": "10px"}
        ),
    ]
),

        # Network Visualization (NetworkX)
        html.Div([
    html.H3("Interactive Network Graph: How I2M Groups Connect Through Co-authorship",
            style={"color": "#2c3e50", "marginBottom": "20px"}),

    html.P("This interactive network reveals the web of collaborations between I2M research groups. Each node is a team; each connection is built on co-authored publications. The thicker the edge, the stronger the partnership. You can drag the nodes to rearrange the structure. This visual map gives life to the internal dynamics of scientific exchange — showing not just who publishes, but who publishes together.",
           style={"fontSize": "16px", "lineHeight": "1.8", "color": "#2c3e50"}),

    html.Button("Display Network", id="show-network-btn", n_clicks=0,
                style={"marginBottom": "20px", "backgroundColor": "#e74c3c", "color": "white", "border": "none", "padding": "10px 20px", "borderRadius": "8px"}),

    cyto.Cytoscape(
        id='group-cytoscape-network',
        elements=[],  # vide au départ
        layout={'name': 'cose'},
        style={'width': '100%', 'height': '600px', 'transition': 'opacity 2s ease-in-out'},
        stylesheet=[
            {
                'selector': 'node',
                'style': {
                    'label': 'data(label)',
                    'background-color': '#e74c3c',
                    'color': 'white',
                    'font-size': '14px',
                    'text-valign': 'center',
                    'text-halign': 'center',
                    'width': '40px',
                    'height': '40px',
                    'border-color': '#e74c3c',
                    'border-width': 2
                }
            },
            {
                'selector': 'edge',
                'style': {
                    'label': '',
                    'width': 'mapData(weight, 1, 35, 2, 12)',
                    'line-color': '#95a5a6',
                    'curve-style': 'bezier'
                }
            },
            {
                'selector': 'edge:hover',
                'style': {
                    'label': 'data(tooltip)',
                    'text-opacity': 1,
                    'text-background-color': '#ecf0f1',
                    'text-background-opacity': 1,
                    'text-background-shape': 'roundrectangle',
                    'font-size': '12px',
                    'color': '#2c3e50',
                    'text-border-color': '#7f8c8d',
                    'text-border-width': 1
                }
            }
        ]
    ),

    html.P("This view highlights key collaboration hubs within I2M. For instance, the strong link between AA and ALEA (31 co-publications) stands out, along with tight connections between AA–AGLR (27) and ALEA–GDAC (24). These partnerships illustrate how some groups play a central role in the institute’s collaborative ecosystem, weaving together multiple research directions through shared work.",
           style={"fontSize": "16px", "lineHeight": "1.8", "color": "#2c3e50"})
], style={
    "backgroundColor": "#ffffff",
    "padding": "30px",
    "marginTop": "40px",
    "marginBottom": "40px",
    "borderLeft": "6px solid #e74c3c",
    "boxShadow": "0px 2px 8px rgba(0,0,0,0.05)",
    "borderRadius": "10px",
    "maxWidth": "900px",
    "margin": "auto"
}),


        # Global Network Metrics
        html.Div(
    style={
        "backgroundColor": "#ffffff",
        "padding": "30px",
        "marginTop": "40px",
        "marginBottom": "40px",
        "borderLeft": "6px solid #16a085",
        "boxShadow": "0px 2px 8px rgba(0,0,0,0.05)",
        "borderRadius": "10px",
        "maxWidth": "900px",
        "margin": "auto"
    },
    children=[
        html.H3("How Strong Is the Web? Unveiling the Global Structure of I2M’s Network",
                style={"color": "#2c3e50", "marginBottom": "20px"}),

        html.P(
            "Behind every collaboration lies a structure. To better understand how I2M research groups connect as a whole, "
            "we explore key global network metrics. These indicators help us grasp the strength, cohesion, and efficiency of the internal scientific web. "
            "Select a metric below to reveal what the numbers say about how tightly the institute is woven together.",
            style={"fontSize": "16px", "color": "#2c3e50"}
        ),

        dcc.Dropdown(
            id='metric-selector',
            options=[{"label": k, "value": k} for k in global_metrics.keys()],
            placeholder="Choose a metric...",
            style={"width": "100%", "marginBottom": "20px"}
        ),

        html.Div(id='metric-output', style={
            "backgroundColor": "#fdfefe",
            "padding": "25px",
            "fontSize": "18px",
            "color": "#1a5276",
            "borderRadius": "10px",
            "border": "2px dashed #1abc9c",
            "textAlign": "center",
            "boxShadow": "0px 2px 8px rgba(0,0,0,0.03)"
        }),

        
    ]
),

        # Node-Level Metrics
       html.Div(
    style={
        "backgroundColor": "#ffffff",
        "padding": "30px",
        "marginTop": "40px",
        "marginBottom": "40px",
        "borderLeft": "6px solid #f39c12",
        "boxShadow": "0px 2px 8px rgba(0,0,0,0.05)",
        "borderRadius": "12px",
        "maxWidth": "900px",
        "margin": "auto"
    },
    children=[
        html.H3("Node-Level Metrics: Who Are the Central Characters in I2M’s Story?",
                style={"color": "#2c3e50", "marginBottom": "20px"}),

        html.P("In this interactive section, explore the network roles played by each research group. Are they bridges, hubs, or universally connected nodes? "
               "Select a group, then a metric, to uncover the meaning behind their position in the network.",
               style={"fontSize": "16px", "color": "#2c3e50", "lineHeight": "1.8"}),

        html.Div([
            html.Div([
                html.Label("Choose a research group:", style={"marginBottom": "5px"}),
                dcc.Dropdown(
                    id='centrality-group-selector',
                    options=[{"label": group, "value": group} for group in centrality_full_df.index],
                    placeholder="Select a group...",
                    style={"width": "100%"}
                )
            ], style={"width": "48%", "display": "inline-block", "marginRight": "4%"}),

            html.Div([
                html.Label("Choose a centrality metric:", style={"marginBottom": "5px"}),
                dcc.Dropdown(
                    id='centrality-metric-selector',
                    options=[{"label": metric, "value": metric} for metric in centrality_full_df.columns],
                    placeholder="Select a metric...",
                    style={"width": "100%"}
                )
            ], style={"width": "48%", "display": "inline-block"})
        ], style={"marginBottom": "25px"}),

        html.Div(id='centrality-output-box', style={
            "backgroundColor": "#fdfefe",
            "padding": "25px",
            "fontSize": "18px",
            "color": "#1a5276",
            "borderRadius": "10px",
            "border": "2px dashed #f39c12",
            "textAlign": "center",
            "boxShadow": "0px 2px 8px rgba(0,0,0,0.03)"
        })
    ]
),








        # Community Detection
       html.Div([
    html.H3("Community Detection: Mapping the Hidden Structure of Collaboration",
            style={"color": "#2c3e50", "marginBottom": "20px"}),

    html.P("The Louvain algorithm helps reveal how I2M research groups self-organize into clusters. "
           "Each cluster represents a group of research teams more tightly connected internally than externally. "
           "This community structure offers a new lens on I2M’s scientific landscape — uncovering sub-networks that may reflect shared research themes or collaborative traditions.",
           style={"fontSize": "16px", "color": "#2c3e50", "lineHeight": "1.8"}),

    html.Button("Display Community Network", id="show-community-network", n_clicks=0,
                style={"marginBottom": "20px", "backgroundColor": "#8e44ad", "color": "white",
                       "border": "none", "padding": "10px 20px", "borderRadius": "8px"}),

    cyto.Cytoscape(
        id='cytoscape-communities',
        elements=[],
        layout={'name': 'cose'},
        style={'width': '100%', 'height': '600px', 'transition': 'opacity 2s ease-in-out'},
        stylesheet=[
            {
                'selector': 'node',
                'style': {
                    'label': 'data(label)',
                    'color': '#2c3e50',
                    'text-valign': 'center',
                    'text-halign': 'center',
                    'width': 50,
                    'height': 50,
                    'font-size': '14px'
                }
            },
            *[{
                'selector': f'.comm-{i}',
                'style': {'background-color': colors[i % len(colors)]}
            } for i in unique_communities],
            {
                'selector': 'edge',
                'style': {
                    'width': 'mapData(weight, 1, 35, 2, 12)',
                    'line-color': '#bbb'
                }
            },
            {
                'selector': 'edge:hover',
                'style': {
                    'label': 'data(tooltip)',
                    'font-size': '12px',
                    'text-background-color': '#fefefe',
                    'text-background-opacity': 1,
                    'color': '#2c3e50'
                }
            }
        ]
    ),

    html.Br(),
    html.Label("Select a community to explore:", style={"fontSize": "16px", "color": "#2c3e50"}),

    dcc.Dropdown(
        id='community-selector',
        options=[{"label": f"Community {i}", "value": i} for i in unique_communities],
        placeholder="Choose a community...",
        style={"marginBottom": "20px"}
    ),

    html.Div(id='community-info-box', style={
        "backgroundColor": "#fdfefe",
        "padding": "25px",
        "fontSize": "17px",
        "color": "#2c3e50",
        "borderRadius": "10px",
        "border": "2px dashed #8e44ad",
        "textAlign": "left",
        "boxShadow": "0px 2px 8px rgba(0,0,0,0.03)"
    })
], style={
    "backgroundColor": "#ffffff",
    "padding": "30px",
    "marginTop": "40px",
    "marginBottom": "40px",
    "borderLeft": "6px solid #8e44ad",
    "boxShadow": "0px 2px 8px rgba(0,0,0,0.05)",
    "borderRadius": "12px",
    "maxWidth": "900px",
    "margin": "auto"
}),




        # Assortativity and Clustering Coefficient
        html.Div(
    style={
        "backgroundColor": "#ffffff",
        "padding": "30px",
        "marginTop": "40px",
        "marginBottom": "40px",
        "borderLeft": "6px solid #3498db",
        "boxShadow": "0px 2px 8px rgba(0,0,0,0.05)",
        "borderRadius": "10px",
        "maxWidth": "900px",
        "margin": "auto"
    },
    children=[
        html.H3(" Assortativity and Clustering: Reading the Collective Behavior",
                style={"color": "#2c3e50", "marginBottom": "20px"}),

        html.P("To complement our exploration of network structure, we examine two global indicators that reveal hidden patterns in I2M’s collaboration web: "
               "assortativity and clustering. Together, they help us understand whether groups tend to collaborate with similar peers, and whether tight-knit "
               "triads form across the network.",
               style={"fontSize": "16px", "color": "#2c3e50", "lineHeight": "1.8"}),

        dcc.RadioItems(
            id='network-metric-choice',
            options=[
                {'label': 'Assortativity', 'value': 'Assortativity'},
                {'label': 'Global Clustering Coefficient', 'value': 'Global Clustering Coefficient'}
            ],
            value='Assortativity',
            labelStyle={'display': 'inline-block', 'marginRight': '20px'},
            style={"marginBottom": "20px"}
        ),

        html.Div(id='network-metric-output', style={
            "backgroundColor": "#fdfefe",
            "padding": "25px",
            "fontSize": "17px",
            "color": "#2c3e50",
            "borderRadius": "10px",
            "border": "2px dashed #3498db",
            "textAlign": "left",
            "boxShadow": "0px 2px 8px rgba(0,0,0,0.03)"
        })
    ]
),

        # Qualitative Roles of Research Groups
        html.Div(
    style={
        "backgroundColor": "#ffffff",
        "padding": "30px",
        "marginTop": "40px",
        "marginBottom": "0px",
        "borderLeft": "6px solid #e67e22",
        "boxShadow": "0px 2px 8px rgba(0,0,0,0.05)",
        "borderRadius": "10px",
        "maxWidth": "900px",
        "margin": "auto"
    },
    children=[
        html.H3("What Role Does Each Group Play?",
                style={"color": "#2c3e50", "marginBottom": "20px"}),

        html.P("Each research group plays a different role in the I2M collaboration network. Some act as bridges between disconnected teams, "
               "others form strong local clusters, and some serve as connectors with widespread ties. In this section, we explore those roles "
               "and what they reveal about how I2M collaborates internally.",
               style={"fontSize": "16px", "lineHeight": "1.8", "color": "#2c3e50"}),

        dcc.Dropdown(
            id='group-role-selector',
            options=[{"label": group, "value": group} for group in centrality_df.index],
            placeholder="Choose a research group...",
            style={"marginBottom": "20px"}
        ),

        html.Div(id='role-output-box', style={
            "backgroundColor": "#fdfefe",
            "padding": "25px",
            "fontSize": "17px",
            "color": "#2c3e50",
            "borderRadius": "10px",
            "border": "2px dashed #e67e22",
            "textAlign": "left",
            "boxShadow": "0px 2px 8px rgba(0,0,0,0.03)"
        })
    ]
),


        # Temporal Evolution of Group Collaboration
        html.Div(
    style={
        "backgroundColor": "#ffffff",
        "padding": "30px",
        "marginTop": "40px",
        "marginBottom": "40px",
        "borderLeft": "6px solid #e74c3c",
        "boxShadow": "0px 2px 8px rgba(0,0,0,0.05)",
        "borderRadius": "10px",
        "maxWidth": "900px",
        "margin": "auto"
    },
    children=[
        html.H3("From Pairs to Networks: The Rising Tide of Internal Collaboration at I2M", style={"color": "#2c3e50"}),

        html.P("How have collaborations evolved inside I2M over the past decade?"
               "To answer this, we divide time into three key periods, allowing us to observe how relationships between research groups have transformed — from isolated partnerships to a dense web of interactions. This journey reveals not just when collaborations happened, but how they grew in scale, complexity, and cohesion."),

        dcc.Dropdown(
            id="temporal-period-selector",
            options=[{"label": key, "value": key} for key in temporal_networks.keys()],
            placeholder="Select a period...",
            style={"marginTop": "20px", "marginBottom": "20px"}
        ),

        cyto.Cytoscape(
            id="temporal-collab-network",
            layout={"name": "cose"},
            style={"width": "100%", "height": "600px"},
            elements=[],
            stylesheet=[
                {
                    "selector": "node",
                    "style": {
                        "label": "data(label)",
                        "background-color": "#e74c3c",
                        "color": "#2c3e50",
                        "text-valign": "center",
                        "text-halign": "center",
                        "width": "40px",
                        "height": "40px",
                        "font-size": "14px"
                    }
                },
                {
                    "selector": "edge",
                    "style": {
                        "width": "mapData(weight, 1, 20, 2, 12)",
                        "line-color": "#95a5a6",
                        "curve-style": "bezier"
                    }
                },
                {
                    "selector": "edge:hover",
                    "style": {
                        "label": "data(tooltip)",
                        "font-size": "12px",
                        "color": "#2c3e50",
                        "text-background-color": "#f9f9f9",
                        "text-background-opacity": 1
                    }
                }
            ]
        ),

        html.Div(id="temporal-network-description", style={
    "backgroundColor": "#fdfefe",
    "padding": "30px",
    "fontSize": "18px",
    "lineHeight": "1.8",
    "color": "#2c3e50",
    "borderRadius": "10px",
    "border": "2px dashed #e74c3c",
    "textAlign": "left",
    "boxShadow": "0px 2px 8px rgba(0,0,0,0.03)",
    "marginTop": "25px"
}),
        html.P("Over time, I2M’s internal collaborations have evolved from isolated duos to a collective ecosystem."
               " This temporal journey reveals the impact of shared goals, institutional initiatives, and growing interdisciplinary trust — leading to a network that is not only productive, but united.")
    ]
),
        # Transition: From Research Groups to Authors
html.Div(
    style={
        "backgroundColor": "#ffffff",
        "padding": "30px",
        "marginTop": "40px",
        "marginBottom": "40px",
        "borderLeft": "6px solid #16a085",
        "boxShadow": "0px 2px 8px rgba(0,0,0,0.05)",
        "borderRadius": "10px",
        "maxWidth": "900px",
        "margin": "auto"
    },
    children=[
        html.H3("From Collaboration Networks to Individual Stories",
                style={"color": "#2c3e50", "marginBottom": "20px"}),

        html.P("Inside I2M, research groups don't operate in silos — they form a dynamic, interconnected network of scientific exchange. What began as a data-cleaning step evolved into a detailed exploration of internal collaborations, uncovering which groups frequently co-author publications and how these relationships evolve over time.",
               style={"fontSize": "16px", "lineHeight": "1.8", "color": "#2c3e50"}),

        html.P("Key partnerships like AA–ALEA and ALEA–GDAC form the backbone of this network. The graph revealed a tightly knit structure where no group is isolated and all are within immediate reach — a rare and valuable level of cohesion. Network metrics confirmed this density and proximity, illustrating that collaboration at I2M is not just present, but deeply embedded.",
               style={"fontSize": "16px", "lineHeight": "1.8", "color": "#2c3e50"}),

        html.P("On a local scale, hubs like AA and ALEA emerged as influential, while AGT stood out as a bridge between otherwise separate teams. Community detection highlighted natural clusters — likely shaped by research themes or shared history. Viewed over time, the network evolved from isolated pairings to a strong, interconnected core, reflecting a maturing collaborative culture.",
               style={"fontSize": "16px", "lineHeight": "1.8", "color": "#2c3e50"}),


        html.P("Yet, beyond group-level patterns lie the individuals — the authors whose scientific paths, collaborations, and decisions truly shape the network.",
               style={"fontSize": "16px", "lineHeight": "1.8", "color": "#2c3e50"}),

        html.P("Who are the most central figures? Which researchers act as bridges across domains or groups? Are collaborations driven by research topics, or personal connections?",
               style={"fontSize": "16px", "lineHeight": "1.8", "color": "#2c3e50"}),

        html.P("To answer these questions, we now shift our focus to the author level — zooming in on the people behind the publications and the intricate micro-dynamics that drive I2M’s scientific momentum.",
               style={"fontSize": "16px", "lineHeight": "1.8", "color": "#2c3e50"})
    ]
),

        # 2.2 Collaboration Between Authors¶
html.Div(
    style={
        "backgroundColor": "#ffffff",
        "padding": "30px",
        "marginTop": "60px",
        "marginBottom": "20px",
        "borderLeft": "6px solid #f39c12",  
        "boxShadow": "0px 2px 10px rgba(0,0,0,0.04)",
        "borderRadius": "10px",
        "maxWidth": "920px",
        "margin": "auto"
    },
    children=[
        html.H3("2.2 Collaboration Between Authors: Mapping I2M’s Internal Scientific Network",
                style={"color": "#2c3e50", "marginBottom": "10px"}),

        html.P("While group-level collaboration gives us a structured overview, the true scientific heartbeat lies in individual connections. In this section, we turn our lens toward I2M researchers themselves — revealing who collaborates with whom, how frequently, and what patterns emerge when we follow co-authorship at the author level. From central hubs to isolated contributors, this analysis highlights the people powering the institute’s scientific output.",
               style={"fontSize": "15px", "lineHeight": "1.8", "color": "#2c3e50"})
    ]
),

        # Author Collaboration Network Construction
html.Div(
    style={
        "backgroundColor": "#ffffff",
        "padding": "30px",
        "marginTop": "40px",
        "marginBottom": "40px",
        "borderLeft": "6px solid #8e44ad",
        "boxShadow": "0px 2px 8px rgba(0,0,0,0.04)",
        "borderRadius": "10px",
        "maxWidth": "900px",
        "margin": "auto"
    },
    children=[
        html.H3("Zoom In on the Human Side of Research: Discover Who Collaborates With Whom",
                style={"color": "#2c3e50", "marginBottom": "20px"}),

       html.P(
    "Behind every publication lies a story of collaboration — researchers exchanging ideas, "
    "solving problems, and building something together. "
    "This interactive graph brings these human connections to life. "
    "Search for any I2M author to reveal their personal collaboration network: who they work with, "
    "how densely they’re connected, and how central they are in the broader ecosystem. "
    "You can zoom, drag, and hover to explore individual ties, co-publication counts, and "
    "the hidden architecture of teamwork within the lab. "
    "Whether you're looking to understand a key researcher’s role or to spot isolated clusters and central hubs, "
    "this visualization transforms names into a living, breathing map of collaboration.",
    style={"fontSize": "16px", "lineHeight": "1.8", "color": "#2c3e50"}
),

        html.Div([
            html.Label("Search an author:", style={"fontSize": "16px"}),
            dcc.Dropdown(
                id="author-selector",
                options=[{"label": a, "value": a} for a in all_author_names],
                placeholder="Type an author's name...",
                style={"marginBottom": "20px"}
            )
        ]),

        cyto.Cytoscape(
    id="author-network-graph",
    elements=[],  
    layout={'name': 'cose'},
    style={'width': '100%', 'height': '600px'},
    stylesheet=[
    {
        'selector': 'node',
        'style': {
            'label': 'data(label)',
            'background-color': '#f06292',
            'color': 'black',
            'font-size': '12px',
            'text-valign': 'center',
            'text-halign': 'center',
            'text-wrap': 'wrap',  
            'text-max-width': '80px', 
            'width': 30,
            'height': 30
        }
    },
    {
        'selector': 'edge',
        'style': {
            'width': 'mapData(weight, 1, 30, 1, 10)',
            'line-color': '#b2babb',
            'curve-style': 'bezier'
        }
    },
    {
        'selector': 'edge:hover',
        'style': {
            'label': 'data(tooltip)',
            'font-size': '10px',
            'color': '#2c3e50',
            'text-background-color': '#fefefe',
            'text-background-opacity': 1,
            'text-border-color': '#aaa',
            'text-border-width': 1
        }
    },
    {
        'selector': '.highlighted',
        'style': {
            'background-color': '#8e44ad',
            'width': 40,
            'height': 40,
            'font-size': '15px'
        }
    }
]

)

    ]
),
        # Core Network Metrics
        html.Div(
    style={
        "backgroundColor": "#ffffff",
        "padding": "30px",
        "marginTop": "40px",
        "marginBottom": "40px",
        "borderLeft": "6px solid #3498db",
        "boxShadow": "0px 2px 8px rgba(0,0,0,0.05)",
        "borderRadius": "10px",
        "maxWidth": "900px",
        "margin": "auto"
    },
    children=[
        html.H3("How Cohesive Is the Author Network at I2M?",
                style={"color": "#2c3e50", "marginBottom": "20px"}),

        html.P(
            "Beyond group dynamics, author-level collaboration reveals a different layer of structure. "
            "Here, we explore the overall shape of I2M's author-to-author network: how tightly it is connected, "
            "how fragmented it is, and how each individual fits into the bigger picture.",
            style={"fontSize": "16px", "color": "#2c3e50"}
        ),

        dcc.Dropdown(
            id='author-metric-selector',
            options=[{"label": k, "value": k} for k in author_metrics.keys()],
            placeholder="Choose a metric...",
            style={"width": "100%", "marginBottom": "20px"}
        ),

        html.Div(id='author-metric-output', style={
            "backgroundColor": "#fdfefe",
            "padding": "25px",
            "fontSize": "18px",
            "color": "#1a5276",
            "borderRadius": "10px",
            "border": "2px dashed #3498db",
            "textAlign": "center",
            "boxShadow": "0px 2px 8px rgba(0,0,0,0.03)"
        }),
    ]
),
        # Node-Level Centrality
        html.Div(
    style={
        "backgroundColor": "#ffffff",
        "padding": "30px",
        "marginTop": "40px",
        "marginBottom": "40px",
        "borderLeft": "6px solid #e67e22",
        "boxShadow": "0px 2px 8px rgba(0,0,0,0.05)",
        "borderRadius": "10px",
        "maxWidth": "900px",
        "margin": "auto"
    },
    children=[
        html.H3("Node-Level Centrality: Who Shapes the Network From Within?",
                style={"color": "#2c3e50", "marginBottom": "20px"}),

        html.P("Not all nodes in a network are equal — some act as bridges, connecting distant parts of the graph, while others serve as hubs, attracting many connections and anchoring the structure. To uncover these influential figures within I2M’s internal collaboration network, we explore a series of centrality metrics. These measures help us identify the researchers who play pivotal roles in the flow of knowledge, the ones who connect the dots, hold the structure together, or amplify collaboration across teams.",
               style={"fontSize": "16px", "lineHeight": "1.8", "color": "#2c3e50"}),

        html.Label("Choose an author to explore:", style={"marginBottom": "5px", "fontSize": "16px"}),
        dcc.Dropdown(
            id="author-centrality-selector",
            options=[{"label": a, "value": a} for a in centrality_authors_df["Author"]],
            placeholder="Select an I2M author...",
            style={"marginBottom": "20px"}
        ),

        html.Div(id="author-centrality-output", style={
            "backgroundColor": "#fdfefe",
            "padding": "25px",
            "fontSize": "17px",
            "color": "#2c3e50",
            "borderRadius": "10px",
            "border": "2px dashed #e67e22",
            "textAlign": "left",
            "boxShadow": "0px 2px 8px rgba(0,0,0,0.03)"
        })
    ]
),
        # Identify Key Connectors
        
        html.Div(
    style={
        "backgroundColor": "#ffffff",
        "padding": "30px",
        "marginTop": "40px",
        "marginBottom": "40px",
        "borderLeft": "6px solid #e74c3c",
        "boxShadow": "0px 2px 8px rgba(0,0,0,0.05)",
        "borderRadius": "10px",
        "maxWidth": "900px",
        "margin": "auto"
    },
    children=[
        html.H3("Hidden Bridges: Meet the Authors Linking the I2M Network",
                style={"color": "#2c3e50", "marginBottom": "20px"}),

        html.P("Not all collaborations are created equal. While some researchers work steadily within their own groups, others act as bridges — quietly linking distinct scientific communities, facilitating the flow of knowledge across disciplines. These key connectors often sit on the shortest paths between otherwise distant authors. By identifying them, we reveal the invisible threads that hold the I2M network together. Click below to uncover the individuals who keep the collaboration graph truly connected.",
               style={"fontSize": "16px", "lineHeight": "1.8", "color": "#2c3e50"}),

        html.Button("Click to Reveal Top Bridge Authors", id="show-connectors-button", n_clicks=0,
                    style={"marginTop": "15px", "marginBottom": "20px", "padding": "10px 20px",
                           "backgroundColor": "#e74c3c", "color": "white", "border": "none", "borderRadius": "5px",
                           "cursor": "pointer", "fontWeight": "bold"}),

        html.Div(id="connectors-output", style={
            "marginTop": "30px",
            "display": "flex",
            "flexWrap": "wrap",
            "gap": "20px",
            "justifyContent": "center"
        })
    ]
),

        # Community Detection
        html.Div(
    style={
        "backgroundColor": "#ffffff",
        "padding": "30px",
        "marginTop": "40px",
        "marginBottom": "40px",
        "borderLeft": "6px solid #16a085",
        "boxShadow": "0px 2px 8px rgba(0,0,0,0.05)",
        "borderRadius": "10px",
        "maxWidth": "900px",
        "margin": "auto"
    },
    children=[
        html.H3("Communities Within the Network: Discover Who Sticks Together",
                style={"color": "#2c3e50", "marginBottom": "20px"}),

        html.P("Within the larger I2M network, tight-knit author communities naturally form — groups of researchers who collaborate frequently, share research topics, or maintain long-standing partnerships. These clusters often represent shared expertise, ongoing projects, or deep academic bonds. Use the dropdown to explore each community and understand how it functions internally: Who plays a central role? Who connects this group to others? Who operates on the periphery? Every cluster has its own dynamic — and its own story to tell.",
               style={"fontSize": "16px", "lineHeight": "1.8", "color": "#2c3e50"}),

        html.Label("Select a community:", style={"marginBottom": "5px", "fontSize": "16px"}),

        dcc.Dropdown(
            id="community-selector-unique",
            options=global_community_dropdown_options,
            placeholder="Choose a community...",
            style={"marginBottom": "20px"}
        ),

        html.Div(id="community-output", style={
            "backgroundColor": "#fdfefe",
            "padding": "25px",
            "fontSize": "17px",
            "color": "#2c3e50",
            "borderRadius": "10px",
            "border": "2px dashed #16a085",
            "textAlign": "left",
            "boxShadow": "0px 2px 8px rgba(0,0,0,0.03)"
        })
    ]
),


        # Focused Subgraphs
        html.Div(
    style={
        "backgroundColor": "#ffffff",
        "padding": "30px",
        "marginTop": "40px",
        "marginBottom": "40px",
        "borderLeft": "6px solid #f39c12",
        "boxShadow": "0px 2px 8px rgba(0,0,0,0.05)",
        "borderRadius": "10px",
        "maxWidth": "900px",
        "margin": "auto"
    },
    children=[
        html.H3(" Deep Dive into Inner Circles: Communities and Cliques That Shape I2M",
                style={"color": "#2c3e50", "marginBottom": "20px"}),

        html.P("While broad networks show us the big picture, the real magic often happens in the smaller, tightly-knit circles — the communities and cliques where collaboration is most intense.",
               style={"fontSize": "16px", "color": "#2c3e50"}),

        html.P("In this section, we zoom into those inner circles:", style={"fontSize": "16px", "color": "#2c3e50"}),

        html.P("• Communities reveal thematic alliances and recurring partnerships.",
               style={"fontSize": "16px", "color": "#2c3e50"}),

        html.P("• Cliques represent the most cohesive subteams — where every member works directly with every other.",
               style={"fontSize": "16px", "color": "#2c3e50"}),

        html.P("Use the dropdowns below to explore who belongs to each group and uncover the stories behind their scientific bonds.",
               style={"fontSize": "16px", "color": "#2c3e50"}),

        html.Div([
            html.Label("Select a Louvain Community:", style={"marginBottom": "5px", "fontSize": "16px"}),
            dcc.Dropdown(
                id="community-dropdown",
                options=focused_community_dropdown_options,
                placeholder="Choose a community...",
                style={"marginBottom": "20px"}
            ),
            html.Div(id="community-display", style={
                "backgroundColor": "white",
                "padding": "20px",
                "border": "2px dashed #f39c12",
                "borderRadius": "8px",
                "boxShadow": "none",
                "marginBottom": "30px",
                "fontSize": "15px",
                "color": "#2c3e50",
                "whiteSpace": "pre-wrap"
            })
        ]),

        html.Div([
            html.Label("Select a Clique:", style={"marginBottom": "5px", "fontSize": "16px"}),
            dcc.Dropdown(
                id="clique-dropdown",
                options=clique_dropdown_options,
                placeholder="Choose a clique...",
                style={"marginBottom": "20px"}
            ),
            html.Div(id="clique-display", style={
                "backgroundColor": "white",
                "padding": "20px",
                "border": "2px dashed #f39c12",
                "borderRadius": "8px",
                "boxShadow": "none",
                "fontSize": "15px",
                "color": "#2c3e50",
                "whiteSpace": "pre-wrap"
            })
        ])
    ]
),

        # Temporal Trends of Author Collaborations
       html.Div(
    style={
        "backgroundColor": "#ffffff",
        "padding": "30px",
        "marginTop": "40px",
        "marginBottom": "40px",
        "borderLeft": "6px solid #8e44ad",
        "boxShadow": "0px 2px 8px rgba(0,0,0,0.05)",
        "borderRadius": "10px",
        "maxWidth": "900px",
        "margin": "auto"
    },
    children=[
        html.H3("Tracing the Pulse of Collaboration Over Time",
                style={"color": "#2c3e50", "marginBottom": "20px"}),

        html.P("How has I2M’s internal collaboration network evolved throughout the years? Click on the Play button to explore the unfolding story — year by year — of how authors connected, collaborated, and adapted in the face of global challenges.",
               style={"fontSize": "16px", "color": "#2c3e50"}),

        html.Button("Animate Chart", id="temporal-animate-button", n_clicks=0, style={
            "marginBottom": "20px",
            "backgroundColor": "#8e44ad",
            "color": "white",
            "border": "none",
            "padding": "10px 20px",
            "fontSize": "16px",
            "borderRadius": "5px",
            "cursor": "pointer"
        }),

        dcc.Graph(id="temporal-trend-graph"),

        html.Div(id="temporal-observations", style={
            "marginTop": "20px",
            "fontSize": "15px",
            "color": "#2c3e50",
            "whiteSpace": "pre-wrap"
        })
    ]
),


        # conclusion 2.2
        html.Div(
    style={
        "backgroundColor": "#ffffff",
        "padding": "30px",
        "marginTop": "40px",
        "marginBottom": "40px",
        "borderLeft": "6px solid #3498db",  
        "boxShadow": "0px 2px 8px rgba(0,0,0,0.05)",
        "borderRadius": "10px",
        "maxWidth": "900px",
        "margin": "auto"
    },
    children=[
        html.H3("From Structure to Story: Insights from I2M’s Internal Collaboration Network",
                style={"color": "#2c3e50", "marginBottom": "20px"}),

        html.P("Over the course of this internal exploration, we peeled back the layers of I2M’s co-authorship network — from its foundational structure to the individuals who shape it most.",
               style={"fontSize": "16px", "lineHeight": "1.8", "color": "#2c3e50"}),

        html.P("We began by constructing the collaborative fabric connecting I2M researchers, revealing not just isolated connections but an intricate web of intellectual partnerships. With the help of centrality metrics, we identified key figures who act as bridges, hubs, and influencers — silently orchestrating the flow of ideas within the institute.",
               style={"fontSize": "16px", "lineHeight": "1.8", "color": "#2c3e50"}),

        html.P("We saw that while the network remains relatively sparse, it maintains internal cohesion through a handful of critical connectors and cohesive clusters. Some authors serve as bridges across research boundaries, others anchor tightly-knit groups with intense co-publication activity.",
               style={"fontSize": "16px", "lineHeight": "1.8", "color": "#2c3e50"}),

        html.P("Community detection revealed the thematic or team-based groupings that emerge naturally through collaboration. And by diving into focused subgraphs and cliques, we caught a glimpse of the microcosms that power collective scientific work. Finally, by tracing these patterns over time, we observed how collaboration grew, fractured, and resumed — shaped not only by internal momentum but also by global events such as the COVID-19 pandemic.",
               style={"fontSize": "16px", "lineHeight": "1.8", "color": "#2c3e50"}),

        html.P("In short, the internal collaboration network of I2M is dynamic, structured, and increasingly integrated — supported by key individuals and core research communities.",
               style={"fontSize": "16px", "lineHeight": "1.8", "color": "#2c3e50"}),


        html.P("But science doesn’t happen in isolation.",
               style={"fontSize": "16px", "lineHeight": "1.8", "color": "#2c3e50"}),

        html.P("While these internal networks form the backbone of I2M’s collaborative culture, they represent only part of the picture. To fully understand how I2M researchers connect with the broader scientific community, we now turn to the external dimension.",
               style={"fontSize": "16px", "lineHeight": "1.8", "color": "#2c3e50"}),

        html.P("In the next section, we explore external collaborations — mapping how I2M researchers link with other institutions, identifying external partners, and uncovering patterns of international and interdisciplinary exchange that extend far beyond the walls of the institute.",
               style={"fontSize": "16px", "lineHeight": "1.8", "color": "#2c3e50", "marginBottom": "0px"})
    ]
),
        # ESPACE VISUEL ENTRE LES PARTIES
html.Div(style={"height": "80px"}),  
        
        # 3. External Collaborations Analysis
        html.Div(
    style={
        "backgroundColor": "#ffffff",
        "padding": "30px",
        "marginTop": "50px",
        "marginBottom": "40px",
        "borderLeft": "6px solid #e67e22",  
        "boxShadow": "0px 5px 20px rgba(0,0,0,0.05)",
        "borderRadius": "12px",
        "maxWidth": "950px",
        "margin": "auto"
    },
    children=[
        html.H2("3. External Collaborations Analysis",
                style={"color": "#2c3e50", "marginBottom": "20px"}),

        html.P("After uncovering the internal dynamics that shape I2M’s scientific output, we now broaden our lens. Collaboration within the institute tells only part of the story — the rest unfolds through its connections to the wider research world.",
               style={"fontSize": "16px", "lineHeight": "1.8", "color": "#2c3e50"}),

        html.P("In this section, we explore how I2M researchers engage with peers across other institutions, countries, and disciplines. These external collaborations reflect the institute’s openness, its international reach, and its ability to integrate global expertise into its scientific endeavors.",
               style={"fontSize": "16px", "lineHeight": "1.8", "color": "#2c3e50"}),

        html.P("By analyzing external co-authorships, we aim to identify I2M’s most frequent partners, spot emerging international alliances, and reveal how interdisciplinary exchange shapes the research landscape. This outward-looking perspective complements the internal one — offering a more complete portrait of I2M’s collaborative identity.",
               style={"fontSize": "16px", "lineHeight": "1.8", "color": "#2c3e50"})
    ]
),

       #  Identification of External Co-Authors
        html.Div(
    style={
        "backgroundColor": "#ffffff",
        "padding": "30px",
        "marginTop": "60px",
        "marginBottom": "20px",
        "borderLeft": "6px solid #e74c3c",
        "boxShadow": "0px 2px 10px rgba(0,0,0,0.04)",
        "borderRadius": "10px",
        "maxWidth": "920px",
        "margin": "auto"
    },
    children=[
        html.H3("Mapping the Beyond: Who Are I2M’s External Collaborators?",
                style={"color": "#2c3e50", "marginBottom": "10px"}),

        html.P(
            "Behind every I2M publication lies a story of interaction — not only among internal teams, "
            "but also with a vast web of scientists beyond the lab’s walls. "
            "In this section, we explore just how far I2M’s collaborative reach extends.",
            style={"fontSize": "15px", "lineHeight": "1.8", "color": "#2c3e50"}
        ),

        html.Label("Choose what to display:", style={"fontSize": "16px", "marginTop": "15px"}),

        dcc.Dropdown(
            id="collab-summary-dropdown",
            options=external_dropdown_options,
            value=None,
            placeholder="Select a category to display...",
            style={"marginBottom": "20px"}
        ),

        html.Div(id="collab-summary-output", style={
            "backgroundColor": "#fdfefe",
            "padding": "20px",
            "fontSize": "16px",
            "color": "#2c3e50",
            "borderRadius": "10px",
            "border": "2px dashed #e74c3c",
            "textAlign": "left",
            "boxShadow": "0px 1px 6px rgba(0,0,0,0.02)"
        })
    ]
),


        # Identify Top External Collaborators
        html.Div(
    style={
        "backgroundColor": "#ffffff",
        "padding": "30px",
        "marginTop": "60px",
        "marginBottom": "20px",
        "borderLeft": "6px solid #16a085",
        "boxShadow": "0px 2px 10px rgba(0,0,0,0.04)",
        "borderRadius": "10px",
        "maxWidth": "920px",
        "margin": "auto"
    },
    children=[
        html.H3("Trusted Allies Beyond the Walls: I2M’s Top External Collaborators",
                style={"color": "#2c3e50", "marginBottom": "10px"}),

        html.P("Every enduring scientific relationship leaves a trace — in co-authored papers, shared discoveries, and long-term trust. In this section, we shine a light on the top 20 external collaborators who have partnered most frequently with I2M researchers.",
               style={"fontSize": "15px", "lineHeight": "1.8", "color": "#2c3e50"}),

        html.Button("Click here to display Top External Collaborators",
                    id="btn-show-top-external", n_clicks=0,
                    style={"backgroundColor": "#117864", "color": "white", "padding": "10px 20px", "border": "none",
                           "borderRadius": "6px", "fontSize": "16px", "marginTop": "15px", "cursor": "pointer"}),

        html.Div(id="top-external-output", style={"marginTop": "30px"})
    ]
),
        # Institutions by Co-publication Frequency
        html.Div(
    style={
        "backgroundColor": "#ffffff",
        "padding": "30px",
        "marginTop": "60px",
        "marginBottom": "20px",
        "borderLeft": "6px solid #f39c12",
        "boxShadow": "0px 2px 10px rgba(0,0,0,0.04)",
        "borderRadius": "10px",
        "maxWidth": "920px",
        "margin": "auto"
    },
    children=[
        html.H3("Mapping Trusted Allies: Institutions That Collaborate Most with I2M", style={"color": "#2c3e50", "marginBottom": "10px"}),

        html.P("Scientific collaboration often goes beyond individual researchers — it becomes a bond between institutions. Some universities, research centers, or hospitals appear again and again in I2M’s publications, signaling the presence of deep, long-term scientific relationships. In this section, we highlight the most frequent external affiliations found across all co-authored publications. These institutions aren’t just names on a page — they represent strategic partners, shared labs, and cross-border research stories that define I2M’s broader scientific identity.By exploring these institutional ties, we gain a clearer view of how I2M connects to the world, both nationally and internationally.",
               style={"fontSize": "15px", "lineHeight": "1.8", "color": "#2c3e50"}),

        html.Button(
    "Click here to display Institutions and UMRs that frequently collaborate with I2M",
    id="show-affiliations-btn",
    n_clicks=0,
    style={
        "backgroundColor": "#f39c12",
        "color": "white",
        "padding": "10px 20px",
        "border": "none",
        "borderRadius": "6px",
        "fontSize": "16px",
        "marginTop": "15px",
        "cursor": "pointer"
    }
),


        html.Div(id="affiliation-cards-output")
    ]
),
        # Visualization of External Collaboration Network
        html.Div([
    html.H3("Explore the Web Beyond: Interactive Network of I2M’s External Collaborations",
            style={"color": "#2c3e50", "marginBottom": "20px"}),

    html.P("Step into the network of I2M’s external scientific partnerships. This interactive graph reveals how individual researchers at I2M connect with collaborators beyond the institute. Use the selector to choose an external author and watch their connections unfold — each link tells a story of shared ideas, co-authored work, and cross-institutional exchange. Every node is labeled with the researcher’s name, allowing you to explore their collaborative links clearly.",
           style={"fontSize": "16px", "lineHeight": "1.8", "color": "#2c3e50"}),

    dcc.Dropdown(
        id="external-author-selector",
        options=network_dropdown_options,
        placeholder="Select an external author...",
        style={"width": "100%", "marginBottom": "20px"}
    ),

    cyto.Cytoscape(
    id='external-author-network-graph',
    elements=[],
    layout={'name': 'cose'},
    style={'width': '100%', 'height': '600px'},
    stylesheet=[
        {
            'selector': 'node',
            'style': {
                'label': 'data(label)',
                'text-wrap': 'wrap',
                'text-max-width': 100,
                'background-color': '#f06292',  
                'color': 'black',
                'font-size': '14px',
                'text-valign': 'center',
                'text-halign': 'center',
                'width': '40px',
                'height': '40px',
                'border-color': '#e91e63',
                'border-width': 2
            }
        },
        {
            'selector': '[type="i2m"]',
            'style': {
                'background-color': '#8e44ad',  
                'border-color': '#6c3483'
            }
        },
        {
            'selector': 'edge',
            'style': {
                'width': 'mapData(weight, 1, 20, 2, 12)',
                'line-color': '#bdc3c7',
                'curve-style': 'bezier'
            }
        },
        {
            'selector': 'edge:hover',
            'style': {
                'label': 'data(tooltip)',
                'text-opacity': 1,
                'text-background-color': '#ecf0f1',
                'text-background-opacity': 1,
                'font-size': '12px',
                'color': '#2c3e50',
            }
        }
    ]
),


    html.P("This network isn't just a visual map — it’s a window into real partnerships. By observing how external collaborators are linked to specific I2M researchers, we can spot strong, recurring alliances that span disciplines, projects, and borders. The label below each node shows the name of the researcher — letting the connections speak for themselves. ",
           style={"fontSize": "16px", "lineHeight": "1.8", "color": "#2c3e50"}),
],
style={
    "backgroundColor": "#ffffff",
    "padding": "30px",
    "marginTop": "60px",
    "marginBottom": "20px",
    "borderLeft": "6px solid #8e44ad",
    "boxShadow": "0px 2px 10px rgba(0,0,0,0.05)",
    "borderRadius": "10px",
    "maxWidth": "920px",
    "margin": "auto"
}),
        # Focus on Strategic Collaborations (UMRs at AMU)
        html.Div([
    html.H3("Zoom on Strategic Collaborations within AMU",
            style={"color": "#2c3e50", "marginBottom": "20px"}),

    html.P("Which labs at Aix-Marseille University collaborate the most with I2M? By scanning the affiliations, we identified a handful of research units within AMU that have co-authored publications with I2M. Click the button below to reveal the most frequent internal university collaborations.",
           style={"fontSize": "16px", "lineHeight": "1.8", "color": "#2c3e50"}),

    html.Button("Reveal Top AMU Collaborations", id="show-amu-collabs-btn", n_clicks=0,
                style={"marginBottom": "20px", "backgroundColor": "#3498db", "color": "white", "border": "none",
                       "padding": "10px 20px", "borderRadius": "5px", "cursor": "pointer"}),

    dcc.Graph(id="amu-collab-barplot", style={"height": "600px"}),

    html.P("We refined our analysis by focusing on affiliations that explicitly mention 'aix-marseille' or 'amu', along with several known variations. This allowed us to better identify strategic internal collaborations within Aix-Marseille University. Only a few institutions matched these criteria. These results suggest that while internal AMU collaborations exist, they are relatively rare in this dataset and are likely underrepresented in the affiliation metadata. This highlights the need for more accurate and standardized affiliation reporting in scientific databases.",
           style={"fontSize": "16px", "lineHeight": "1.8", "color": "#2c3e50"})
],
style={
    "backgroundColor": "#ffffff",
    "padding": "30px",
    "marginTop": "60px",
    "marginBottom": "20px",
    "borderLeft": "6px solid #3498db",
    "boxShadow": "0px 2px 10px rgba(0,0,0,0.05)",
    "borderRadius": "10px",
    "maxWidth": "920px",
    "margin": "auto"
}),

        # Conclusion de la partie 3
        html.Div(
    style={
        "backgroundColor": "#ffffff",
        "padding": "30px",
        "marginTop": "60px",
        "marginBottom": "20px",
        "borderLeft": "6px solid #e67e22",
        "boxShadow": "0px 2px 10px rgba(0,0,0,0.05)",
        "borderRadius": "10px",
        "maxWidth": "920px",
        "margin": "auto"
    },
    children=[
        html.H3("A Structural View of Scientific Connectivity",
                style={"color": "#2c3e50", "marginBottom": "20px"}),

        html.P("Through this analysis, we’ve uncovered the vast external footprint of I2M researchers. Starting with a high-level count of external co-authors, we discovered a striking number of 2,855 unique collaborators — far outnumbering the internal pool of I2M authors. This highlights the lab’s strong orientation toward openness and collaboration beyond institutional walls.",
               style={"fontSize": "16px", "lineHeight": "1.8", "color": "#2c3e50"}),

        html.P("By zooming in, we identified the most frequent external partners — individuals who appear consistently across co-authored publications. These names hint at long-term scientific partnerships, often built across countries and disciplines. We then stepped back to look at the institutions behind those names: major French UMRs, international research centers, and academic powerhouses.",
               style={"fontSize": "16px", "lineHeight": "1.8", "color": "#2c3e50"}),

        html.P("The interactive network visualization offered a structural view of these external ties, revealing not only the central roles played by some I2M researchers but also distinct clusters of collaboration, where small scientific ecosystems seem to thrive.",
               style={"fontSize": "16px", "lineHeight": "1.8", "color": "#2c3e50"}),

        html.P("Finally, we focused on strategic internal partnerships, particularly those within Aix-Marseille University (AMU). Despite I2M’s location within AMU, internal collaborations with other UMRs remain relatively rare in the metadata — opening up opportunities to strengthen institutional synergy in the future.",
               style={"fontSize": "16px", "lineHeight": "1.8", "color": "#2c3e50"}),


        html.P("While we now understand who collaborates with I2M — both internally and externally — a deeper question remains: what are they collaborating on?",
               style={"fontSize": "16px", "lineHeight": "1.8", "color": "#2c3e50"}),

        html.P("To answer this, we now turn to the content itself: the scientific topics, domains, and research themes explored by I2M’s researchers.",
               style={"fontSize": "16px", "lineHeight": "1.8", "color": "#2c3e50"})
    ]
),

         # ESPACE VISUEL ENTRE LES PARTIES
html.Div(style={"height": "80px"}),  # espace vertical de 80px

        # 4. Thematic and Scientific Domain Analysis
        html.Div(
    style={
        "backgroundColor": "#ffffff",
        "padding": "30px",
        "marginTop": "50px",
        "marginBottom": "40px",
        "borderLeft": "6px solid #e74c3c",
        "boxShadow": "0px 5px 20px rgba(0,0,0,0.05)",
        "borderRadius": "10px",
        "maxWidth": "950px",
        "margin": "auto"
    },
    children=[
        html.H2("4. Thematic and Scientific Domain Analysis",
                style={"color": "#2c3e50", "fontSize": "28px", "marginBottom": "15px"}),

        html.P("Having mapped out I2M’s collaborative landscape, we now shift our attention to the content of its research. After all, scientific partnerships are built not only on institutional ties but on shared questions, disciplines, and missions.",
               style={"fontSize": "16px", "lineHeight": "1.8", "color": "#2c3e50"}),

        html.P("What are the dominant research themes within I2M’s output? Which topics unite different teams, and which signal emerging priorities? By exploring abstracts, titles, and keywords, we aim to trace the scientific DNA of the lab — its core areas of expertise, its evolution over time, and the diversity of its contributions.",
               style={"fontSize": "16px", "lineHeight": "1.8", "color": "#2c3e50"}),

        html.P("In this next chapter, we dive into a thematic dissection of I2M’s scientific publications.",
               style={"fontSize": "16px", "lineHeight": "1.8", "color": "#2c3e50"})
    ]
),

        # Keyword Extraction from Titles

html.Div([
    html.H3("Words That Speak: What Titles Reveal About I2M’s Scientific Identity",
            style={"color": "#2c3e50", "marginBottom": "20px"}),

    html.P("Every research title carries a fragment of the story. In this section, we extract the most frequent keywords from I2M publication titles — "
           "giving us a unique window into the lab’s core research directions. "
           "Use the input below to adjust how many terms you want to see in the word cloud.",
           style={"fontSize": "16px", "lineHeight": "1.8", "color": "#2c3e50"}),

    html.Div([
        dcc.Input(id='wordcloud-topk', type='number', value=100, min=10, max=200, step=10,
                  placeholder="Number of top keywords",
                  style={"padding": "8px", "marginRight": "10px", "borderRadius": "5px",
                         "border": "1px solid #ccc", "width": "150px"}),

        html.Button("Generate WordCloud", id='generate-wordcloud-btn', n_clicks=0,
                    style={"backgroundColor": "#16a085", "color": "white", "border": "none",
                           "padding": "10px 20px", "borderRadius": "5px", "cursor": "pointer"})
    ], style={"marginBottom": "20px"}),

    html.Img(id='wordcloud-img', style={"width": "100%", "borderRadius": "5px"}),

    html.P("From the cloud of words, dominant themes begin to emerge — painting a picture of I2M’s research DNA:",
           style={"fontSize": "16px", "marginTop": "30px", "color": "#2c3e50"}),

    html.Ul([
        html.Li([html.B("Mathematical modeling"), " — Keywords like model, equation, and theory suggest strong theoretical foundations."]),
        html.Li([html.B("Numerical methods"), " — Terms such as finite, codes, and approach highlight computational techniques."]),
        html.Li([html.B("Stochastic or probabilistic models"), " — The frequent use of ‘random’ points to uncertainty modeling and probabilistic thinking."]),
        html.Li([html.B("Mathematical structures and spaces"), " — Words like groups, spaces, and fields reveal abstract frameworks in use."]),
        html.Li([html.B("Applications to dynamic systems and flows"), " — Recurring terms like dynamics, flows, and functions reflect applied directions."])
    ], style={"fontSize": "16px", "lineHeight": "1.8", "color": "#2c3e50"}),

    html.P("Together, these terms reflect a lab deeply engaged in both theoretical and applied mathematics, "
           "numerical analysis, and the study of complex systems.",
           style={"fontSize": "16px", "lineHeight": "1.8", "color": "#2c3e50"}),

   
],
style={
    "backgroundColor": "#ffffff",
    "padding": "30px",
    "marginTop": "60px",
    "marginBottom": "20px",
    "borderLeft": "6px solid #16a085",
    "boxShadow": "0px 2px 10px rgba(0,0,0,0.05)",
    "borderRadius": "10px",
    "maxWidth": "920px",
    "margin": "auto"
}),
        #  Dominant Themes by Research Group
        html.Div([
    html.H3("What Drives Each Group? A Glimpse Into I2M’s Research DNA",
            style={"color": "#2c3e50", "marginBottom": "20px"}),

    html.P("Each I2M research group follows a distinct intellectual path. To uncover their specific scientific focus, select a group from the menu below. You’ll discover a visual word cloud of their most frequent keywords, along with a brief thematic summary that captures the essence of their research.",
           style={"fontSize": "16px", "lineHeight": "1.8", "color": "#2c3e50"}),

    dcc.Dropdown(
        id="group-selector",
        options=[{"label": g, "value": g} for g in group_wordclouds.keys()],
        placeholder="Select a research group",
        style={"marginBottom": "20px"}
    ),

    html.Img(id="group-wordcloud", style={"width": "100%", "borderRadius": "5px", "marginBottom": "20px"}),

    html.Div(id="group-description", style={"fontSize": "16px", "lineHeight": "1.8", "color": "#2c3e50"})
],
style={
    "backgroundColor": "#ffffff",
    "padding": "30px",
    "marginTop": "60px",
    "marginBottom": "20px",
    "borderLeft": "6px solid #f39c12",
    "boxShadow": "0px 2px 10px rgba(0,0,0,0.05)",
    "borderRadius": "10px",
    "maxWidth": "920px",
    "margin": "auto"
}),
        # Evolution of Scientific Topics Over Time
        html.Div([
    html.H3("Tracing the Evolution of Scientific Focus at I2M",
            style={"color": "#2c3e50", "marginBottom": "20px"}),

    html.P(
        "What scientific questions have captured I2M’s attention over the years? To uncover this, we examined "
        "publication titles year by year and extracted the most frequent keywords. Select a year below to explore "
        "the changing landscape of topics, from enduring foundations to emerging themes.",
        style={"fontSize": "16px", "lineHeight": "1.8", "color": "#2c3e50"}),

    dcc.Dropdown(
        id="year-selector",
        options=[{"label": str(y), "value": y} for y in sorted(year_wordclouds.keys())],
        placeholder="Select a year",
        style={"marginBottom": "20px"}
    ),

    html.Img(id="year-wordcloud", style={"width": "100%", "borderRadius": "5px", "marginBottom": "20px"}),

    html.Div([
        html.Ul([
            html.Li([
                html.Span("Enduring Foundations: ", style={"fontWeight": "bold"}),
                "Keywords such as model, equations, and random appear consistently across the years. "
                "These reflect I2M's deep-rooted engagement with mathematical modeling, partial differential equations, and stochastic processes — a scientific backbone that stands the test of time."
            ]),
            html.Li([
                html.Span("Shifting Priorities: ", style={"fontWeight": "bold"}),
                "Starting in 2019, terms like codes and finite began to appear more frequently, signaling a growing momentum in fields like coding theory and discrete mathematics. "
                "Meanwhile, groups and dynamics surged after 2020, highlighting interest in algebraic structures and dynamic systems."
            ]),
            html.Li([
                html.Span("Sudden Highlights: ", style={"fontWeight": "bold"}),
                "Certain keywords experienced temporary bursts — measures and infinite rose significantly in 2016, while homogeneous and clustering appeared as new focal points in 2024."
            ]),
            
        ], style={"lineHeight": "1.8", "fontSize": "16px", "color": "#2c3e50"})
    ])
],
style={
    "backgroundColor": "#ffffff",
    "padding": "30px",
    "marginTop": "60px",
    "marginBottom": "20px",
    "borderLeft": "6px solid #8e44ad",
    "boxShadow": "0px 2px 10px rgba(0,0,0,0.05)",
    "borderRadius": "10px",
    "maxWidth": "920px",
    "margin": "auto"
}),

        # Conclusion de la partie 4
        html.Div(
    style={
        "backgroundColor": "#ffffff",
        "padding": "30px",
        "marginTop": "60px",
        "marginBottom": "20px",
        "borderLeft": "6px solid #3498db",
        "boxShadow": "0px 2px 10px rgba(0,0,0,0.05)",
        "borderRadius": "10px",
        "maxWidth": "920px",
        "margin": "auto"
    },
    children=[
        html.H3("Unveiling the Scientific Identity of I2M",
                style={"color": "#2c3e50", "marginBottom": "20px"}),

        html.P("As we delved into the scientific content of I2M’s publications, a rich and multifaceted thematic landscape emerged. From global patterns in publication titles to the specific interests of each research group, we uncovered the intellectual DNA that defines the laboratory.",
               style={"fontSize": "16px", "lineHeight": "1.8", "color": "#2c3e50"}),

        html.P("Through keyword extraction, we saw how I2M's research is grounded in core mathematical pillars such as modeling, equations, stochastic processes, and numerical methods. Yet, we also witnessed signs of evolution: new terms gaining momentum, others fading, and research interests branching into emerging or interdisciplinary directions.",
               style={"fontSize": "16px", "lineHeight": "1.8", "color": "#2c3e50"}),

        html.P("Each research group exhibited its own thematic identity — some rooted in topology and geometry, others in probability, coding theory, or dynamical systems — highlighting the lab’s diversity in expertise and approach.",
               style={"fontSize": "16px", "lineHeight": "1.8", "color": "#2c3e50"}),

        html.P("Beyond numbers and networks, these findings reveal a dynamic research ecosystem, constantly shaped by both long-standing traditions and new scientific challenges.",
               style={"fontSize": "16px", "lineHeight": "1.8", "color": "#2c3e50"})
    ]
)











       

       





        
       




        




       







        
        

        

        





         















       
    ]
)

# I2M
@app.callback(
    Output("group-description-box", "children"),
    Input("group-dropdown", "value")
)
def update_group_description(selected_group):
    if selected_group:
        return group_descriptions[selected_group]
    return ""




# publications per year
@callback(
    Output('pubs-bar-graph', 'figure'),
    Input('animate-button', 'n_clicks')
)
def animate_bars(n_clicks):
    if n_clicks == 0:
        raise dash.exceptions.PreventUpdate

    x = pubs_per_year_df['Year']
    y = pubs_per_year_df['Number of Publications']

    fig = go.Figure(
        data=[go.Bar(
            x=x,
            y=[0]*len(y),
            marker_color='#16a085',
            hovertemplate='Year: %{x}<br>Number of Publications: %{y}<extra></extra>'
        )],
        layout=go.Layout(
            title='Number of Publications per Year',
            xaxis=dict(title='Year', tickangle=45),
            yaxis=dict(title='Number of Publications', range=[0, max(y) + 10]),
            plot_bgcolor='white',
            paper_bgcolor='white',
            title_font_size=20,
            margin=dict(t=60, b=40, l=20, r=20),
            updatemenus=[{
                'type': 'buttons',
                'showactive': False,
                'buttons': [{
                    'label': 'Play',
                    'method': 'animate',
                    'args': [None, {
                        'frame': {'duration': 100, 'redraw': True},
                        'transition': {'duration': 200, 'easing': 'cubic-in-out'},
                        'fromcurrent': True,
                        'mode': 'immediate'
                    }]
                }]
            }]
        ),
        frames=[
            go.Frame(
                data=[go.Bar(
                    x=x,
                    y=(y * i / 10).round(0),
                    marker_color='#16a085',
                    hovertemplate='Year: %{x}<br>Number of Publications: %{y}<extra></extra>'
                )],
                name=str(i)
            )
            for i in range(1, 11)
        ]
    )

    return fig

# document type distribution
@callback(
    Output('doc-bar-graph', 'figure'),
    Input('animate-doc-button', 'n_clicks')
)
def animate_doc_type_chart(n_clicks):
    if n_clicks == 0:
        raise dash.exceptions.PreventUpdate

    x = doc_type_df['Count']
    y = doc_type_df['Document Type']
    percent = doc_type_df['Percent']

    fig = go.Figure(
        data=[go.Bar(
            x=[0]*len(x),  # Commence à 0
            y=y,
            orientation='h',
            marker_color='#d98880',
            hovertemplate='Document Type: %{y}<br>Count: %{x} (%{customdata}%)<extra></extra>',
            customdata=percent
        )],
        layout=go.Layout(
            title='Distribution of Document Types',
            xaxis=dict(title='Count', range=[0, max(x) + 50]),
            yaxis=dict(title='Document Type', categoryorder='total ascending'),
            plot_bgcolor='white',
            paper_bgcolor='white',
            title_font_size=20,
            margin=dict(t=60, b=40, l=20, r=20),
            updatemenus=[{
                'type': 'buttons',
                'showactive': False,
                'buttons': [{
                    'label': 'Play',
                    'method': 'animate',
                    'args': [None, {
                        'frame': {'duration': 100, 'redraw': True},
                        'transition': {'duration': 200, 'easing': 'cubic-in-out'},
                        'fromcurrent': True,
                        'mode': 'immediate'
                    }]
                }]
            }]
        ),
        frames=[
            go.Frame(
                data=[go.Bar(
                    x=(x * i / 10).round(0),  # Allonge progressivement la longueur
                    y=y,
                    orientation='h',
                    marker_color='#8e44ad',
                    hovertemplate='Document Type: %{y}<br>Count: %{x} (%{customdata}%)<extra></extra>',
                    customdata=percent
                )],
                name=str(i)
            )
            for i in range(1, 11)
        ]
    )

    return fig

# Publications by Research Group
@callback(
    Output('group-bar-graph', 'figure'),
    Input('animate-group-button', 'n_clicks')
)
def animate_group_chart(n_clicks):
    if n_clicks == 0:
        raise dash.exceptions.PreventUpdate

    x = group_df['Number of Publications']
    y = group_df['Research Group']

    fig = go.Figure(
        data=[go.Bar(
            x=[0]*len(x),
            y=y,
            orientation='h',
            marker_color='#a569bd',
            hovertemplate='Group: %{y}<br>Publications: %{x}<extra></extra>'
        )],
        layout=go.Layout(
            title="Number of Publications by Research Group",
            xaxis=dict(title='Number of Publications', range=[0, max(x) + 50]),
            yaxis=dict(title='Research Group', categoryorder='total ascending'),
            plot_bgcolor='white',
            paper_bgcolor='white',
            title_font_size=20,
            margin=dict(t=60, b=40, l=20, r=20),
            updatemenus=[{
                'type': 'buttons',
                'showactive': False,
                'buttons': [{
                    'label': 'Play',
                    'method': 'animate',
                    'args': [None, {
                        'frame': {'duration': 100, 'redraw': True},
                        'transition': {'duration': 200, 'easing': 'cubic-in-out'},
                        'fromcurrent': True,
                        'mode': 'immediate'
                    }]
                }]
            }]
        ),
        frames=[
            go.Frame(
                data=[go.Bar(
                    x=(x * i / 10).round(0),
                    y=y,
                    orientation='h',
                    marker_color='#3498db',
                    hovertemplate='Group: %{y}<br>Publications: %{x}<extra></extra>'
                )],
                name=str(i)
            )
            for i in range(1, 11)
        ]
    )

    return fig

# Publications by Author
@callback(
    Output('author-bar-graph', 'figure'),
    Input('animate-author-button', 'n_clicks')
)
def animate_author_chart(n_clicks):
    if n_clicks == 0:
        raise dash.exceptions.PreventUpdate

    x = author_df['Number of Publications']
    y = author_df['Author']

    fig = go.Figure(
        data=[go.Bar(
            x=[0]*len(x),
            y=y,
            orientation='h',
            marker_color='#5dade2',
            hovertemplate='Author: %{y}<br>Publications: %{x}<extra></extra>'
        )],
        layout=go.Layout(
            title='Top 20 Most Prolific I2M Authors',
            xaxis=dict(title='Number of Publications', range=[0, max(x) + 10]),
            yaxis=dict(title='Author', categoryorder='total ascending'),
            plot_bgcolor='white',
            paper_bgcolor='white',
            title_font_size=20,
            margin=dict(t=60, b=40, l=20, r=20),
            updatemenus=[{
                'type': 'buttons',
                'showactive': False,
                'buttons': [{
                    'label': 'Play',
                    'method': 'animate',
                    'args': [None, {
                        'frame': {'duration': 100, 'redraw': True},
                        'transition': {'duration': 200, 'easing': 'cubic-in-out'},
                        'fromcurrent': True,
                        'mode': 'immediate'
                    }]
                }]
            }]
        ),
        frames=[
            go.Frame(
                data=[go.Bar(
                    x=(x * i / 10).round(0),
                    y=y,
                    orientation='h',
                    marker_color='#e67e22',
                    hovertemplate='Author: %{y}<br>Publications: %{x}<extra></extra>'
                )],
                name=str(i)
            )
            for i in range(1, 11)
        ]
    )

    return fig

# Internal vs External Collaborations: rien

# Collaboration Intensity: Internal vs External
@callback(
    Output('collab-bar-graph', 'figure'),
    Input('animate-collab-button', 'n_clicks')
)
def animate_collab_chart(n_clicks):
    if n_clicks == 0:
        raise dash.exceptions.PreventUpdate

    x = collab_stats['Collaboration Type']
    y = collab_stats['Number of Publications']
    percent = collab_stats['Percent']
    color = '#16a085'  # ✅ Emerald Green

    fig = go.Figure(
        data=[go.Bar(
            x=x,
            y=[0]*len(y),
            marker_color=color,
            customdata=percent,
            hovertemplate='Type: %{x}<br>Publications: %{y} (%{customdata}%)<extra></extra>'
        )],
        layout=go.Layout(
            title='Collaboration Types Across Publications',
            xaxis=dict(title='Collaboration Type'),
            yaxis=dict(title='Number of Publications', range=[0, max(y) + 50]),
            plot_bgcolor='white',
            paper_bgcolor='white',
            title_font_size=20,
            margin=dict(t=60, b=40, l=20, r=20),
            updatemenus=[{
                'type': 'buttons',
                'showactive': False,
                'buttons': [{
                    'label': 'Play',
                    'method': 'animate',
                    'args': [None, {
                        'frame': {'duration': 100, 'redraw': True},
                        'transition': {'duration': 200, 'easing': 'cubic-in-out'},
                        'fromcurrent': True,
                        'mode': 'immediate'
                    }]
                }]
            }]
        ),
        frames=[
            go.Frame(
                data=[go.Bar(
                    x=x,
                    y=(y * i / 10).round(0),
                    marker_color=color,
                    customdata=percent,
                    hovertemplate='Type: %{x}<br>Publications: %{y} (%{customdata}%)<extra></extra>'
                )],
                name=str(i)
            )
            for i in range(1, 11)
        ]
    )

    return fig











# Co-publication Matrix (Group to Group)

@app.callback(
    Output("co-pub-result", "children"),
    [Input("group1-dropdown", "value"),
     Input("group2-dropdown", "value")]
)
def update_copub_result(group1, group2):
    if group1 is None or group2 is None:
        return ""
    if group1 == group2:
        return "⚠️ Please select two different groups."

    row = group_edges_df[
        ((group_edges_df['Group 1'] == group1) & (group_edges_df['Group 2'] == group2)) |
        ((group_edges_df['Group 1'] == group2) & (group_edges_df['Group 2'] == group1))
    ]

    if not row.empty:
        weight = row['Weight'].values[0]
        return f"{weight} co-authored publications between {group1} and {group2}."
    else:
        return f"🚫 No recorded collaboration between {group1} and {group2}."

# Network Visualization (NetworkX)
@app.callback(
    Output("group-cytoscape-network", "elements"),
    Input("show-network-btn", "n_clicks")
)
def update_network(n_clicks):
    if n_clicks > 0:
        return generate_network_elements()
    return []


# Global Network Metrics

@app.callback(
    Output("metric-output", "children"),
    Input("metric-selector", "value")
)
def display_selected_metric(metric_name):
    if metric_name is None:
        return ""

    value = global_metrics.get(metric_name, "N/A")
    explanation = global_metric_explanations.get(metric_name, "")
    
    return html.Div([
        html.H4(f"{metric_name} = {value}",
                style={"color": "#1a5276", "marginBottom": "10px", "fontSize": "22px"}),

        html.P(explanation, style={
            "fontSize": "16px",
            "lineHeight": "1.7",
            "color": "#2c3e50"
        })
    ])

# Node-Level Metrics

@app.callback(
    Output("centrality-output-box", "children"),
    [Input("centrality-group-selector", "value"),
     Input("centrality-metric-selector", "value")]
)
def display_personalized_centrality(group, metric):
    if group is None or metric is None:
        return ""

    value = centrality_full_df.loc[group, metric]
    story = centrality_stories.get(group, {}).get(metric, "No interpretation available for this selection.")

    return html.Div([
        html.H4(f"{group} – {metric} = {value:.4f}",
                style={"color": "#1a5276", "marginBottom": "10px", "fontSize": "22px"}),

        html.P(story, style={
            "fontSize": "16px",
            "lineHeight": "1.7",
            "color": "#2c3e50"
        })
    ])











# Community Detection
@app.callback(
    Output("cytoscape-communities", "elements"),
    Input("show-community-network", "n_clicks")
)
def update_community_network(n_clicks):
    if n_clicks > 0:
        return generate_cyto_community_elements()
    return []

@app.callback(
    Output("community-info-box", "children"),
    Input("community-selector", "value")
)
def display_group_community_info(comm):
    if comm is None:
        return html.P(" ")



    members = community_members.get(comm, [])
    text = f"Community {comm} includes the following research groups: " + ", ".join(members) + "."

    if comm == 0:
        insight = "This cluster includes AA and AGLR, suggesting strong bilateral ties or thematic alignment between them."
    elif comm == 1:
        insight = "This group — ALEA, AGT, and GDAC — shows tight internal collaboration, likely across interdisciplinary topics."
    else:
        insight = "This community structure might reflect a sub-network based on specific joint projects or historical ties."

    return html.Div([
        html.P(text, style={"marginBottom": "10px"}),
        html.P(insight)
    ])
# Assortativity and Clustering
@app.callback(
    Output("network-metric-output", "children"),
    Input("network-metric-choice", "value")
)
def update_network_structure_metric(metric_name):
    if metric_name is None:
        return ""

    value = network_structure_metrics[metric_name]["value"]
    explanation = network_structure_metrics[metric_name]["interpretation"]

    value_display = "Not defined (NaN)" if pd.isna(value) else f"{value:.4f}"

    return html.Div([
        html.H4(f"{metric_name} = {value_display}",
                style={"color": "#3498db", "marginBottom": "15px"}),

        html.P(explanation)
    ])

#Roles
@app.callback(
    Output("role-output-box", "children"),
    Input("group-role-selector", "value")
)
def display_group_role(group):
    if group is None:
        return ""

    row = centrality_df.loc[group]
    role = row['Role']
    explanation = role_explanations.get(role, "No explanation available for this role.")

    return html.Div([
        html.H4(f"{group} — Role: {role}", style={"color": "#e67e22", "marginBottom": "10px"}),

        html.P(f"Degree: {row['Degree']}, Betweenness: {row['Betweenness']:.4f}, Clustering: {row['Clustering']:.4f}",
               style={"fontSize": "15px", "marginBottom": "15px"}),

        html.P(explanation)
    ])

# Temporal Evolution of Group Collaboration
@app.callback(
    [Output("temporal-collab-network", "elements"),
     Output("temporal-network-description", "children")],
    Input("temporal-period-selector", "value")
)
def update_temporal_network(period_label):
    if period_label is None:
        return [], ""

    elements = temporal_networks.get(period_label, [])
    explanation = temporal_explanations.get(period_label, "No description available.")
    
    return elements, explanation

# Author Collaboration Network Construction

@app.callback(
    Output("author-network-graph", "elements"),
    Input("author-selector", "value")
)
def update_author_network(selected_author):
    if not selected_author:
        return []

    neighborhood = list(G_authors.neighbors(selected_author))
    sub_elements = []

    for node in [selected_author] + neighborhood:
        sub_elements.append({
            'data': {'id': node, 'label': node},
            'classes': 'highlighted' if node == selected_author else ''
        })

    for u, v, d in G_authors.edges(data=True):
        if (u == selected_author and v in neighborhood) or (v == selected_author and u in neighborhood):
            sub_elements.append({
                'data': {
                    'source': u,
                    'target': v,
                    'weight': d['weight'],
                    'tooltip': d['weight']
                }
            })

    return sub_elements






@app.callback(
    Output("author-metric-output", "children"),
    Input("author-metric-selector", "value")
)
def display_author_metric(metric_name):
    if metric_name is None:
        return ""

    value = author_metrics.get(metric_name, "N/A")
    explanation = author_metric_explanations.get(metric_name, "")

    return html.Div([
        html.H4(f"{metric_name} = {value}",
                style={"color": "#1a5276", "marginBottom": "10px", "fontSize": "22px"}),

        html.P(explanation, style={
            "fontSize": "16px",
            "lineHeight": "1.7",
            "color": "#2c3e50"
        })
    ])

@app.callback(
    Output("author-centrality-output", "children"),
    Input("author-centrality-selector", "value")
)
def display_author_centralities(author):
    if author is None:
        return ""

    row = centrality_authors_df[centrality_authors_df['Author'] == author].iloc[0]
    role = row['Role']

    interpretation = {
        'Bridge': f"{author} acts as a strategic bridge in the network — connecting different areas of research.",
        'Hub': f"{author} is widely connected across I2M, showing a strong collaborative footprint.",
        'Influencer': f"{author} is deeply connected to other influential authors — a key player in spreading ideas.",
        'Peripheral': f"{author} participates with fewer links — a more localized or independent research role."
    }.get(role, "No specific role identified.")

    return html.Div([
    html.H4(f"Author: {author}", style={"color": "#d35400", "marginBottom": "10px"}),

    html.Ul([
        html.Li(f"Degree Centrality: {row['Degree']:.4f}"),
        html.Li(f"Betweenness Centrality: {row['Betweenness']:.4f}"),
        html.Li(f"Closeness Centrality: {row['Closeness']:.4f}"),
        html.Li(f"PageRank: {row['PageRank']:.4f}"),
        html.Li(f"Eigenvector Centrality: {row['Eigenvector']:.4f}"),
        html.Li(f"Assigned Role: {role}")
    ]),
    html.P(interpretation, style={"marginTop": "15px"})
])


#key connectors 
@app.callback(
    Output("connectors-output", "children"),
    Input("show-connectors-button", "n_clicks")
)
def display_connectors(n_clicks):
    if n_clicks == 0:
        return []

    cards = []
    for _, row in top_connectors.iterrows():
        author = row["Author"]
       

        cards.append(html.Div([
    html.H4(author, style={"color": "#2c3e50", "marginBottom": "5px"}),
    html.P(f"Betweenness: {row['Betweenness']:.5f}", style={"margin": "0", "marginBottom": "8px"}),
    html.P(f"Role: {row['Role']}", style={"margin": "0"})
], style={
    "backgroundColor": "#fefefe",
    "padding": "15px",
    "borderLeft": "6px solid #e74c3c",
    "borderRadius": "8px",
    "width": "250px",
    "boxShadow": "0px 2px 8px rgba(0,0,0,0.1)"
}))


    return cards


# Community Detection among Authors

@app.callback(
    Output("community-output", "children"),
    Input("community-selector-unique", "value")
)
def update_global_author_output(selected_community):
    if selected_community is None:
        return html.P(" ")

    subset = global_community_centrality_df[
        global_community_centrality_df["Community"] == selected_community
    ].sort_values(by="Betweenness", ascending=False)

    if subset.empty:
        return html.P("No data available for this community.", style={"color": "red"})

    return html.Div([
        html.H4(f"Authors in Community {selected_community}:"),
        html.Ul([
            html.Li(f"{row['Author']} — Betweenness: {row['Betweenness']:.4f}, Role: {row['Role']}")
            for _, row in subset.iterrows()
        ])
    ])


from dash import Input, Output

from dash import Input, Output

# Focused Subgraph
@app.callback(
    Output("community-display", "children"),
    Input("community-dropdown", "value")
)
def display_selected_community(selected_index):
    if selected_index is None:
        return " "
    
    members = focused_community_members_lookup[selected_index]
    interpretation = detect_interpretation_community(members)

    return html.Div([
        html.P(f"Members of Community {selected_index + 1}:", style={"fontWeight": "bold"}),
        html.P("\n".join(members)),
        html.Hr(),
        html.P(interpretation)
    ])

@app.callback(
    Output("clique-display", "children"),
    Input("clique-dropdown", "value")
)
def display_selected_clique(selected_index):
    if selected_index is None:
        return " "
    
    members = clique_members_lookup[selected_index]
    interpretation = clique_interpretations.get(selected_index, "No interpretation available.")

    return html.Div([
        html.P(f"Members of Clique {selected_index + 1}:", style={"fontWeight": "bold"}),
        html.P("\n".join(members)),
        html.Hr(),
        html.P(interpretation)
    ])


from dash import Output, Input
import plotly.graph_objects as go

# Temporal Trends of Author Collaborations
@callback(
    [Output("temporal-trend-graph", "figure"),
     Output("temporal-observations", "children")],
    Input("temporal-animate-button", "n_clicks")
)
def display_temporal_graph(n_clicks):
    if n_clicks == 0:
        raise dash.exceptions.PreventUpdate

    animation_frames = []

    for i in range(1, len(evolution_df) + 1):
        frame = go.Frame(
            data=[
                go.Scatter(x=evolution_df["Year"][:i], y=evolution_df["Number of Authors"][:i],
                           mode='lines+markers', name='Number of Authors', line=dict(color="#1f77b4")),
                go.Scatter(x=evolution_df["Year"][:i], y=evolution_df["Number of Co-authorships"][:i],
                           mode='lines+markers', name='Number of Co-authorships', line=dict(color="#ff7f0e")),
                go.Scatter(x=evolution_df["Year"][:i], y=evolution_df["Graph Density (%)"][:i],
                           mode='lines+markers', name='Graph Density (%)',
                           line=dict(color="#2ca02c"), yaxis="y2"),
            ],
            name=str(evolution_df["Year"].iloc[i - 1])
        )
        animation_frames.append(frame)

    fig = go.Figure(
        data=[
            go.Scatter(x=[], y=[], mode='lines+markers', name='Number of Authors', line=dict(color="#1f77b4")),
            go.Scatter(x=[], y=[], mode='lines+markers', name='Number of Co-authorships', line=dict(color="#ff7f0e")),
            go.Scatter(x=[], y=[], mode='lines+markers', name='Graph Density (%)',
                       line=dict(color="#2ca02c"), yaxis="y2"),
        ],
        layout=go.Layout(
            title="Evolution of Internal Collaboration Network Over Time",
            xaxis=dict(title="Year"),
            yaxis=dict(title="Authors & Co-authorships"),
            yaxis2=dict(
                title="Graph Density (%)",
                overlaying='y',
                side='right',
                showgrid=False
            ),
            updatemenus=[{
                "type": "buttons",
                "buttons": [{
                    "label": "Play",
                    "method": "animate",
                    "args": [None, {
                        "frame": {"duration": 500, "redraw": True},
                        "fromcurrent": True,
                        "transition": {"duration": 300, "easing": "linear"}
                    }]
                }]
            }],
            plot_bgcolor="white",
            paper_bgcolor="white",
            font=dict(color="#2c3e50"),
            margin=dict(t=60, b=40, l=40, r=20)
        ),
        frames=animation_frames
    )

    return fig, temporal_observations




@app.callback(
    Output("collab-summary-output", "children"),
    Input("collab-summary-dropdown", "value")
)
def display_external_info(selected_stat):
    if selected_stat is None:
        return html.P(" ",
                      style={"fontStyle": "italic", "color": "#7f8c8d", "fontSize": "15px"})

    stat = external_stats[selected_stat]
    return html.Div([
        html.P(f"{selected_stat}: {stat['value']:,}", style={"fontWeight": "bold", "marginBottom": "10px"}),
        html.P(stat["interpretation"], style={"marginTop": "10px"})
    ])

# # Identify Top External Collaborators

@app.callback(
    Output("top-external-output", "children"),
    Input("btn-show-top-external", "n_clicks")
)
def show_top_external_collaborators(n_clicks):
    if n_clicks == 0:
        return ""
    
    cards = generate_external_cards(top_external_collaborators)

    interpretation = html.Div([
        html.P(" This table reveals I2M’s top 20 external collaborators — those who appear most often as co-authors alongside our researchers. Next to each name, you’ll also find their main institutional affiliation, offering insight into where these scientific connections originate and how far I2M’s network extends across the research world.",
               style={"fontWeight": "bold", "marginTop": "10px"}),
        html.P("These repeated collaborations are more than just numbers — they point to solid, enduring partnerships. They reflect the trust, continuity, and shared goals that bind I2M researchers to their external counterparts across institutions and disciplines.",
               style={"marginTop": "5px", "color": "#2c3e50"})
    ], style={"marginBottom": "25px"})

    return [interpretation] + cards

from dash import Output, Input

@app.callback(
    Output("affiliation-cards-output", "children"),
    Input("show-affiliations-btn", "n_clicks")
)
def display_affiliations(n):
    if n > 0:
        return generate_affiliation_cards(top_affiliations_cleaned)
    return []

# Visualization of External Collaboration Network

@app.callback(
    Output("external-author-network-graph", "elements"),
    Input("external-author-selector", "value")
)
def update_external_network(selected_author):
    if not selected_author:
        return []

    neighborhood = list(G_external.neighbors(selected_author)) + [selected_author]
    sub_elements = []

    for node in neighborhood:
        node_type = 'i2m' if node in df['I2M Authors'].explode().unique() else 'external'
        label = node

        sub_elements.append({
            'data': {'id': node, 'label': label, 'type': node_type}
        })

    for u, v, d in G_external.edges(data=True):
        if u in neighborhood and v in neighborhood:
            sub_elements.append({
                'data': {
                    'source': u,
                    'target': v,
                    'weight': d['weight'],
                    'tooltip': d['weight']
                }
            })

    return sub_elements



from dash.dependencies import Input, Output
import plotly.graph_objects as go

# Focus on Strategic Collaborations (UMRs at AMU)
@app.callback(
    Output("amu-collab-barplot", "figure"),
    Input("show-amu-collabs-btn", "n_clicks")
)
def plot_amu_collaborations(n_clicks):
    if n_clicks == 0:
        return go.Figure()

    x = amu_top_counts.values[::-1]
    y = amu_labels.values[::-1]

    fig = go.Figure(
        data=[go.Bar(
            x=[0]*len(x),
            y=y,
            orientation='h',
            marker_color='#f39c12',  
            hovertemplate='%{y}<br>Publications: %{x}<extra></extra>'
        )],
        layout=go.Layout(
            title="Top AMU UMR Collaborations with I2M",
            xaxis_title="Number of Co-authored Publications",
            yaxis_title="AMU Affiliations",
            margin=dict(l=150, r=40, t=60, b=60),
            height=600,
            plot_bgcolor='white',
            xaxis=dict(showgrid=True, gridcolor='lightgrey', zeroline=False),
            yaxis=dict(tickfont=dict(size=11)),
            updatemenus=[{
                'type': 'buttons',
                'showactive': False,
                'buttons': [{
                    'label': 'Play',
                    'method': 'animate',
                    'args': [None, {
                        'frame': {'duration': 100, 'redraw': True},
                        'transition': {'duration': 200, 'easing': 'cubic-in-out'},
                        'fromcurrent': True,
                        'mode': 'immediate'
                    }]
                }]
            }]
        ),
        frames=[
            go.Frame(
                data=[go.Bar(
                    x=(x * i / 10).round(0),
                    y=y,
                    orientation='h',
                    marker_color='#3498db',
                    hovertemplate='%{y}<br>Publications: %{x}<extra></extra>'
                )],
                name=str(i)
            )
            for i in range(1, 11)
        ]
    )

    return fig



# APRES LE LAYOUT

from dash.dependencies import Input, Output

from dash import no_update

@app.callback(
    Output('wordcloud-img', 'src'),
    Input('generate-wordcloud-btn', 'n_clicks'),
    Input('wordcloud-topk', 'value')
)
def update_wordcloud(n_clicks, top_k):
    if n_clicks == 0 or top_k is None:
        return None  # or use no_update
    encoded_img = generate_wordcloud_base64(top_k)
    return f'data:image/png;base64,{encoded_img}'


from dash.dependencies import Input, Output

@app.callback(
    [Output("group-wordcloud", "src"),
     Output("group-description", "children")],
    [Input("group-selector", "value")]
)
def update_group_wordcloud(selected_group):
    print("Selected group:", selected_group)  # ➤ ajoute cette ligne

    if selected_group and selected_group in group_wordclouds:
        img = f"data:image/png;base64,{group_wordclouds[selected_group]}"
        text = group_texts.get(selected_group, "")
        return img, text
    return "", ""

@app.callback(
    Output("year-wordcloud", "src"),
    Input("year-selector", "value")
)
def update_year_wordcloud(selected_year):
    if selected_year and selected_year in year_wordclouds:
        return f"data:image/png;base64,{year_wordclouds[selected_year]}"
    return ""














































        
if __name__ == "__main__":
    app.run(host='0.0.0.0', port=8052, debug=True)








