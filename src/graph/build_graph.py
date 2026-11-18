from pathlib import Path
import pandas as pd
import networkx as nx


def build_account_device_graph(df):
    """Build account-device bipartite graph using training-period rows only."""
    graph = nx.Graph()
    account_col = "card1" if "card1" in df.columns else "nameOrig"
    device_col = "DeviceID" if "DeviceID" in df.columns else "nameDest"
    label_col = "isFraud" if "isFraud" in df.columns else "isFraud"
    for _, row in df.iterrows():
        account = f"account:{row[account_col]}"
        device = f"device:{row[device_col]}"
        graph.add_node(account, bipartite="account")
        graph.add_node(device, bipartite="device")
        graph.add_edge(account, device)
        if label_col in df.columns:
            graph.nodes[account]["fraud"] = max(
                graph.nodes[account].get("fraud", 0), int(row[label_col])
            )
    return graph


def detect_rings(graph):
    suspicious = nx.Graph()
    suspicious.add_nodes_from(
        (node, attrs) for node, attrs in graph.nodes(data=True)
        if node.startswith("account:") and attrs.get("fraud", 0) == 1
    )
    account_nodes = list(suspicious.nodes())
    for left in account_nodes:
        devices = set(graph.neighbors(left))
        for right in account_nodes:
            if left != right and devices.intersection(graph.neighbors(right)):
                suspicious.add_edge(left, right)
    rings = []
    for ring_id, component in enumerate(nx.connected_components(suspicious)):
        if len(component) >= 2:
            for account in component:
                rings.append({"account": account, "ring_id": ring_id, "ring_size": len(component)})
    return pd.DataFrame(rings)


def save_graph_outputs(df, ring_output="reports/rings.csv", plot_output="reports/largest_ring.png"):
    graph = build_account_device_graph(df)
    rings = detect_rings(graph)
    ring_output = Path(ring_output)
    ring_output.parent.mkdir(parents=True, exist_ok=True)
    rings.to_csv(ring_output, index=False)
    return graph, rings
