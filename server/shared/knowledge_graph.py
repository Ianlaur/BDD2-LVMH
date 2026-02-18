"""
Knowledge Graph Builder - Creates organized multi-level graph structure.

This module builds a hierarchical knowledge graph from:
- Taxonomy buckets (top level categories)
- Concepts (from lexicon)
- Client segments (cluster groups)  
- Individual clients (notes)
- Recommended actions

Graph Structure:
    Buckets (6-8 categories)
      └─> Concepts (~75 concepts)
            └─> Clients (400 notes)
                  └─> Segments (8 clusters)
                        └─> Actions (~10 recommendations)

Output: Cytoscape.js compatible JSON for visualization
"""
import json
from pathlib import Path
from typing import Dict, List, Set, Any, Tuple
from collections import defaultdict
import pandas as pd

from server.shared.config import DATA_OUTPUTS, TAXONOMY_DIR
from server.shared.utils import log_stage


class KnowledgeGraphBuilder:
    """Builds an organized knowledge graph with clear hierarchies."""
    
    def __init__(self):
        self.nodes = []
        self.edges = []
        self.node_ids = set()
        self.stats = defaultdict(int)
        
    def add_node(self, node_id: str, label: str, node_type: str, **attrs):
        """Add a node if it doesn't already exist."""
        if node_id in self.node_ids:
            return
        
        self.nodes.append({
            "data": {
                "id": node_id,
                "label": label,
                "type": node_type,
                **attrs
            },
            "classes": node_type
        })
        self.node_ids.add(node_id)
        self.stats[f"nodes_{node_type}"] += 1
    
    def add_edge(self, source: str, target: str, edge_type: str, **attrs):
        """Add an edge between two nodes."""
        edge_id = f"{source}__{edge_type}__{target}"
        
        self.edges.append({
            "data": {
                "id": edge_id,
                "source": source,
                "target": target,
                "type": edge_type,
                **attrs
            },
            "classes": edge_type
        })
        self.stats[f"edges_{edge_type}"] += 1
    
    def build_taxonomy_layer(self, taxonomy: Dict[str, List[str]], 
                            lexicon_df: pd.DataFrame) -> Dict[str, str]:
        """
        Build top-level taxonomy buckets and concepts.
        
        Returns mapping of concept_id -> bucket_name
        """
        log_stage("kg", "Building taxonomy layer...")
        
        concept_to_bucket = {}
        
        # Create bucket nodes (top level categories)
        bucket_order = ["preferences", "intent", "occasion", "constraints", 
                       "lifestyle", "next_action", "other"]
        
        for bucket in bucket_order:
            if bucket not in taxonomy:
                continue
                
            concept_ids = taxonomy[bucket]
            if not concept_ids:
                continue
            
            # Add bucket node
            bucket_id = f"bucket:{bucket}"
            bucket_label = bucket.replace("_", " ").title()
            self.add_node(
                bucket_id, 
                bucket_label,
                "bucket",
                description=f"Category: {bucket}",
                concept_count=len(concept_ids)
            )
            
            # Track concepts in this bucket
            for concept_id in concept_ids:
                concept_to_bucket[concept_id] = bucket
        
        # Create concept nodes and link to buckets
        for _, row in lexicon_df.iterrows():
            concept_id = row['concept_id']
            label = row['label']
            bucket = concept_to_bucket.get(concept_id, 'other')
            
            # Determine if this is a brand/product
            is_brand = bucket in ['preferences'] and any(
                brand_term in label.lower() 
                for brand_term in ['louis vuitton', 'lv', 'dior', 'fendi', 
                                  'givenchy', 'celine', 'kenzo', 'trunk', 
                                  'bag', 'watch', 'jewelry', 'perfume']
            )
            
            concept_node_id = f"concept:{concept_id}"
            self.add_node(
                concept_node_id,
                label,
                "brand" if is_brand else "concept",
                bucket=bucket,
                concept_id=concept_id,
                freq_notes=int(row['freq_notes']) if 'freq_notes' in row else 0,
                languages=row.get('languages', '')
            )
            
            # Link concept to bucket
            bucket_id = f"bucket:{bucket}"
            self.add_edge(
                concept_node_id,
                bucket_id,
                "BELONGS_TO",
                label=f"in category"
            )
        
        log_stage("kg", f"Added {len(bucket_order)} buckets, {len(lexicon_df)} concepts")
        return concept_to_bucket
    
    def build_client_layer(self, note_concepts_df: pd.DataFrame,
                          client_profiles_df: pd.DataFrame,
                          concept_to_bucket: Dict[str, str]):
        """Build client nodes and their connections to concepts."""
        log_stage("kg", "Building client layer...")
        
        # Group concepts by client
        client_concepts = defaultdict(set)
        for _, row in note_concepts_df.iterrows():
            note_id = row['note_id']
            concept_id = row['concept_id']
            client_concepts[note_id].add(concept_id)
        
        # Create client nodes with segment info
        segment_profiles = {}
        for _, row in client_profiles_df.iterrows():
            client_id = row['client_id']  # Changed from note_id
            segment = int(row['cluster_id'])  # Changed from segment
            profile_type = row['profile_type']
            
            # Store segment profile for later
            if segment not in segment_profiles:
                segment_profiles[segment] = {
                    'profile_type': profile_type,
                    'clients': []
                }
            segment_profiles[segment]['clients'].append(client_id)
            
            # Add client node
            client_node_id = f"client:{client_id}"
            self.add_node(
                client_node_id,
                f"Client {client_id}",
                "client",
                client_id=client_id,
                segment=segment,
                profile_type=profile_type,
                concept_count=len(client_concepts.get(client_id, set()))
            )
            
            # Link client to concepts (limit to avoid clutter)
            concepts = list(client_concepts.get(client_id, set()))[:10]  # Top 10
            for concept_id in concepts:
                concept_node_id = f"concept:{concept_id}"
                self.add_edge(
                    client_id,
                    concept_node_id,
                    "INTERESTED_IN",
                    label="interested in"
                )
        
        log_stage("kg", f"Added {len(client_profiles_df)} clients")
        return segment_profiles
    
    def build_segment_layer(self, segment_profiles: Dict[int, Dict]):
        """Build segment nodes grouping similar clients."""
        log_stage("kg", "Building segment layer...")
        
        for segment_id, info in sorted(segment_profiles.items()):
            profile_type = info['profile_type']
            clients = info['clients']
            
            # Create segment node
            segment_node_id = f"segment:{segment_id}"
            self.add_node(
                segment_node_id,
                f"Segment {segment_id}",
                "segment",
                segment_id=segment_id,
                profile_type=profile_type,
                client_count=len(clients)
            )
            
            # Link segment to its clients (sample to avoid clutter)
            sample_clients = clients[:5]  # Show up to 5 representatives
            for client_id in sample_clients:
                client_node_id = f"client:{client_id}"
                self.add_edge(
                    segment_node_id,
                    client_node_id,
                    "INCLUDES",
                    label="includes"
                )
        
        log_stage("kg", f"Added {len(segment_profiles)} segments")
    
    def build_action_layer(self, actions_df: pd.DataFrame):
        """Build action nodes and link to relevant clients."""
        log_stage("kg", "Building action layer...")
        
        # Group actions by client
        client_actions = defaultdict(list)
        for _, row in actions_df.iterrows():
            client_id = row['client_id']  # Changed from note_id
            client_actions[client_id].append(row)
        
        # Track unique actions
        unique_actions = {}
        
        for client_id, actions in client_actions.items():
            client_node_id = f"client:{client_id}"
            
            # Add unique action nodes and link
            for action in actions[:3]:  # Top 3 actions per client
                action_id = action['action_id']
                action_title = action['title']  # Changed from action_title
                
                # Add action node if new
                if action_id not in unique_actions:
                    action_node_id = f"action:{action_id}"
                    self.add_node(
                        action_node_id,
                        action_title,
                        "action",
                        action_id=action_id,
                        channel=action.get('channel', 'Unknown'),
                        priority=action.get('priority', 'Medium')
                    )
                    unique_actions[action_id] = action_node_id
                
                # Link client to action (check if client node exists)
                try:
                    self.add_edge(
                        client_node_id,
                        unique_actions[action_id],
                        "RECOMMENDED",
                        label="recommended"
                    )
                except:
                    pass  # Client node might not exist in filtered view
        
        log_stage("kg", f"Added {len(unique_actions)} unique actions")
    
    def export_cytoscape(self, output_path: Path):
        """Export graph in Cytoscape.js format."""
        cytoscape_data = {
            "elements": self.nodes + self.edges
        }
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(cytoscape_data, f, indent=2, ensure_ascii=False)
        
        log_stage("kg", f"Exported to {output_path}")
        
        # Print stats
        log_stage("kg", "Graph statistics:")
        log_stage("kg", f"  Total nodes: {len(self.nodes)}")
        log_stage("kg", f"  Total edges: {len(self.edges)}")
        for key, value in sorted(self.stats.items()):
            if key.startswith('nodes_'):
                node_type = key.replace('nodes_', '')
                log_stage("kg", f"    - {node_type}: {value}")


def build_knowledge_graph():
    """Main function to build organized knowledge graph."""
    log_stage("kg", "Starting knowledge graph construction...")
    
    # Load required data
    taxonomy_path = TAXONOMY_DIR / "taxonomy_v1.json"
    lexicon_path = TAXONOMY_DIR / "lexicon_v1.json"
    note_concepts_path = DATA_OUTPUTS / "note_concepts.csv"
    client_profiles_path = DATA_OUTPUTS / "client_profiles.csv"
    actions_path = DATA_OUTPUTS / "recommended_actions.csv"
    
    # Check required files
    missing = []
    for path in [taxonomy_path, lexicon_path, note_concepts_path, client_profiles_path]:
        if not path.exists():
            missing.append(path.name)
    
    if missing:
        log_stage("kg", f"ERROR: Missing required files: {', '.join(missing)}")
        return None
    
    # Load data
    with open(taxonomy_path) as f:
        taxonomy = json.load(f)
    
    with open(lexicon_path) as f:
        lexicon_dict = json.load(f)
    
    # Convert lexicon dict to DataFrame
    lexicon_rows = []
    for concept_id, concept_data in lexicon_dict.items():
        lexicon_rows.append({
            'concept_id': concept_id,
            'label': concept_data['label'],
            'freq_notes': concept_data.get('freq_notes', 0),
            'languages': concept_data.get('languages', '')
        })
    lexicon_df = pd.DataFrame(lexicon_rows)
    
    note_concepts_df = pd.read_csv(note_concepts_path)
    client_profiles_df = pd.read_csv(client_profiles_path)
    
    # Build graph
    builder = KnowledgeGraphBuilder()
    
    # Layer 1: Taxonomy (buckets + concepts)
    concept_to_bucket = builder.build_taxonomy_layer(taxonomy, lexicon_df)
    
    # Layer 2: Clients (with concepts)
    segment_profiles = builder.build_client_layer(
        note_concepts_df, 
        client_profiles_df,
        concept_to_bucket
    )
    
    # Layer 3: Segments (grouping clients)
    builder.build_segment_layer(segment_profiles)
    
    # Layer 4: Actions (if available)
    if actions_path.exists():
        actions_df = pd.read_csv(actions_path)
        builder.build_action_layer(actions_df)
    
    # Export
    output_path = DATA_OUTPUTS / "knowledge_graph_cytoscape.json"
    builder.export_cytoscape(output_path)
    
    log_stage("kg", "Knowledge graph construction complete!")
    return output_path


if __name__ == "__main__":
    build_knowledge_graph()
