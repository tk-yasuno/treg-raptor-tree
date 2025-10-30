# 簡易版RAPTOR Tree実行テスト
import json
import numpy as np
import pickle
from pathlib import Path

print("Simple RAPTOR Tree Test")
print("=" * 30)

# データ読み込み
hierarchy_file = "../data/immune_cell_differentiation/immune_cell_hierarchy.json"
with open(hierarchy_file, 'r', encoding='utf-8') as f:
    hierarchy_data = json.load(f)

immune_data = hierarchy_data['immune_cell_hierarchy']
print(f"Loaded {len(immune_data['nodes']) + 1} nodes")

# ダミーベクトル作成
nodes = {}
for i, node_data in enumerate([immune_data['root']] + immune_data['nodes']):
    node_id = node_data['id']
    embedding = np.random.rand(384).astype('float32')
    nodes[node_id] = {
        'data': node_data,
        'embedding': embedding
    }

print(f"Created {len(nodes)} embeddings")

# 保存テスト
output_dir = Path("../data/immune_cell_differentiation/raptor_trees")
output_dir.mkdir(exist_ok=True)

# JSON保存（embedding除外）
tree_data = {
    'nodes': {nid: {**ninfo['data'], 'embedding': None} for nid, ninfo in nodes.items()},
    'metadata': {'test': True, 'nodes_count': len(nodes)}
}

json_file = output_dir / "simple_test.json"
with open(json_file, 'w', encoding='utf-8') as f:
    json.dump(tree_data, f, indent=2, ensure_ascii=False)

print(f"JSON saved: {json_file}")

# Pickle保存
embeddings_data = {nid: ninfo['embedding'] for nid, ninfo in nodes.items()}
pkl_file = output_dir / "simple_test.pkl"
with open(pkl_file, 'wb') as f:
    pickle.dump(embeddings_data, f)

print(f"Embeddings saved: {pkl_file}")

# 読み込み確認
with open(json_file, 'r', encoding='utf-8') as f:
    loaded_data = json.load(f)

with open(pkl_file, 'rb') as f:
    loaded_embeddings = pickle.load(f)

print(f"Load test: {len(loaded_data['nodes'])} nodes, {len(loaded_embeddings)} embeddings")
print("Test completed successfully!")