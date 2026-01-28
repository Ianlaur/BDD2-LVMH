# Knowledge Graph Organization

## What Was Fixed

The knowledge graph has been completely reorganized with a clear hierarchical structure and proper organization.

### Previous Issues
- **Unstructured**: All nodes mixed together without clear organization
- **No hierarchy**: Flat structure made it hard to understand relationships
- **Missing connections**: Concepts, clients, segments not properly linked
- **No visual distinction**: All nodes looked similar
- **Cluttered**: Too many edges creating visual noise

### New Organized Structure

#### 5-Level Hierarchy

```
1. BUCKETS (Top Level - 5-7 categories)
   ├─ preferences (product/brand preferences)
   ├─ intent (emotions, purchase intentions)
   ├─ lifestyle (family, personality, VIP indicators)
   ├─ occasion (events, holidays, life stages)
   ├─ constraints (budget, timing, channels)
   ├─ next_action (appointments, repairs, deliveries)
   └─ other (uncategorized)

2. CONCEPTS (Mid Level - ~20 concepts)
   └─ Linked to their parent bucket
   └─ Includes both brands and generic concepts

3. CLIENTS (Detail Level - 400 clients)
   └─ Each linked to top 10 relevant concepts
   └─ Contains segment and profile information

4. SEGMENTS (Cluster Level - 8 segments)
   └─ Groups clients by similarity
   └─ Shows up to 5 representative clients per segment
   └─ Displays profile type (e.g., "Art Collector | Philanthropist")

5. ACTIONS (Recommendation Level - 3 unique actions)
   └─ Linked to relevant clients
   └─ Shows channel and priority
```

## Visual Organization

### Node Types & Colors
- **🔶 Buckets** (Orange) - Hexagon shape, largest nodes
- **🔴 Brands** (Red) - Diamond shape, mid-size
- **🔵 Concepts** (Blue) - Circle shape, smaller
- **🟢 Clients** (Green) - Small circles
- **🟣 Segments** (Purple) - Rounded rectangles
- **🟠 Actions** (Orange) - Octagon shape

### Edge Types
- `BELONGS_TO` - Concept → Bucket
- `INTERESTED_IN` - Client → Concept
- `INCLUDES` - Segment → Client
- `RECOMMENDED` - Client → Action

## Key Improvements

### 1. Clear Hierarchy
- Top-down organization from categories to individuals
- Easy to understand at any zoom level
- Logical groupings

### 2. Reduced Clutter
- Limited edges per client (top 10 concepts only)
- Sample representatives for segments (5 per segment)
- Top 3 actions per client

### 3. Interactive Features
- **Filter by type**: Show only buckets, concepts, clients, segments, or actions
- **Search**: Highlight matching nodes
- **Layouts**: 5 different layout algorithms
  - Force-Directed (COSE) - Default, physics-based
  - Circle - Circular arrangement
  - Concentric - Rings by importance
  - Grid - Structured rows/columns
  - Hierarchical - Tree-like structure

### 4. Visual Encoding
- **Node size**: Reflects importance (buckets > segments > concepts > clients)
- **Node shape**: Unique shape per type for instant recognition
- **Node color**: Semantic color coding
- **Edge opacity**: Reduced to 50% to avoid visual clutter

## Statistics

Current graph contains:
- **5 Buckets** (top-level categories)
- **20 Concepts** (including brands)
- **400 Clients** (individual customers)
- **8 Segments** (client clusters)
- **3 Actions** (recommendations)
- **2,532 Edges** (relationships)

## Usage

### Dashboard Access
```bash
open client/app/dashboard.html
```

### Regenerate Knowledge Graph
```bash
python -m server.shared.knowledge_graph
```

### Controls
- **Layout dropdown**: Change graph layout algorithm
- **Filter dropdown**: Show specific node types
- **Search box**: Find nodes by name
- **Reset View**: Clear all filters and searches
- **Fit to Screen**: Auto-zoom to fit all visible nodes

## Technical Details

### Files
- **Generator**: `server/shared/knowledge_graph.py`
- **Dashboard**: `server/shared/generate_dashboard.py`
- **Output**: `data/outputs/knowledge_graph_cytoscape.json`
- **Visualization**: `client/app/dashboard.html`

### Integration
The knowledge graph is now part of the standard pipeline:
1. Stage 8: Knowledge Graph Construction
2. Stage 10: Dashboard Generation

Both run automatically when you execute `python -m server.run_all`

## Example Queries

The organized structure enables powerful queries:

1. **"What concepts are in the preferences bucket?"**
   - Filter: Buckets → Select "preferences" node
   - See all connected concept nodes

2. **"Which clients are interested in luxury watches?"**
   - Search: "watch"
   - See all clients connected to watch concept

3. **"What's the profile of Segment 3?"**
   - Filter: Segments → Select "Segment 3"
   - View profile type and representative clients

4. **"What actions are recommended for art collectors?"**
   - Search: "art collector"
   - Follow edges to client nodes
   - See connected action nodes
