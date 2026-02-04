"""
Dashboard Generator - Creates unified HTML dashboard with KG and 3D visualization.

Generates interactive dashboard combining:
- Knowledge Graph (Cytoscape.js)
- 3D Embedding Space (Plotly)
- Client search and filtering
"""
import json
from pathlib import Path
from typing import Dict, Any

from server.shared.config import DATA_OUTPUTS, DEMO_DIR
from server.shared.utils import log_stage


def load_cytoscape_lib() -> str:
    """Load Cytoscape.js library content."""
    # Check if cytoscape.min.js exists in demo folder
    cytoscape_path = DEMO_DIR / "cytoscape.min.js"
    
    if cytoscape_path.exists():
        with open(cytoscape_path, 'r', encoding='utf-8') as f:
            return f.read()
    else:
        # Return CDN fallback
        return ""  # Will use CDN in template


def generate_dashboard():
    """Generate unified dashboard HTML."""
    log_stage("dashboard", "Generating unified dashboard...")
    
    # Load knowledge graph data
    kg_path = DATA_OUTPUTS / "knowledge_graph_cytoscape.json"
    if not kg_path.exists():
        log_stage("dashboard", f"ERROR: Knowledge graph not found: {kg_path}")
        return None
    
    with open(kg_path, 'r', encoding='utf-8') as f:
        kg_data = json.load(f)
    
    # Count stats
    elements = kg_data.get('elements', [])
    nodes = [e for e in elements if 'source' not in e.get('data', {})]
    edges = [e for e in elements if 'source' in e.get('data', {})]
    
    node_types = {}
    for n in nodes:
        t = n['data'].get('type', 'unknown')
        node_types[t] = node_types.get(t, 0) + 1
    
    log_stage("dashboard", f"  Knowledge Graph: {len(nodes)} nodes, {len(edges)} edges")
    for node_type, count in sorted(node_types.items()):
        log_stage("dashboard", f"    - {node_type}: {count}")
    
    # Create HTML dashboard
    html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>LVMH Client Intelligence Dashboard</title>
    <script src="https://cdnjs.cloudflare.com/ajax/libs/cytoscape/3.26.0/cytoscape.min.js"></script>
    <script src="https://cdn.plot.ly/plotly-2.26.0.min.js"></script>
    <style>
        * {{
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }}
        
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: #1a1a1a;
            color: #e0e0e0;
        }}
        
        .header {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            padding: 20px 40px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.3);
        }}
        
        .header h1 {{
            font-size: 28px;
            font-weight: 300;
            letter-spacing: 2px;
        }}
        
        .stats {{
            display: flex;
            gap: 30px;
            margin-top: 15px;
            font-size: 14px;
            opacity: 0.9;
        }}
        
        .stat {{
            display: flex;
            align-items: center;
            gap: 8px;
        }}
        
        .stat-value {{
            font-weight: bold;
            font-size: 18px;
        }}
        
        .tabs {{
            display: flex;
            background: #2a2a2a;
            padding: 0 40px;
            gap: 5px;
        }}
        
        .tab {{
            padding: 15px 30px;
            cursor: pointer;
            border-bottom: 3px solid transparent;
            transition: all 0.3s;
            font-size: 14px;
            font-weight: 500;
            text-transform: uppercase;
            letter-spacing: 1px;
        }}
        
        .tab:hover {{
            background: rgba(255,255,255,0.05);
        }}
        
        .tab.active {{
            border-bottom-color: #667eea;
            background: rgba(255,255,255,0.08);
        }}
        
        .content {{
            display: none;
            height: calc(100vh - 180px);
            padding: 20px 40px;
        }}
        
        .content.active {{
            display: block;
        }}
        
        #kg-container {{
            width: 100%;
            height: 100%;
            background: #252525;
            border-radius: 10px;
            box-shadow: 0 4px 20px rgba(0,0,0,0.3);
        }}
        
        #viz-3d {{
            width: 100%;
            height: 100%;
            border-radius: 10px;
            overflow: hidden;
        }}
        
        .controls {{
            background: #2a2a2a;
            padding: 15px;
            border-radius: 8px;
            margin-bottom: 15px;
            display: flex;
            gap: 15px;
            align-items: center;
        }}
        
        .controls label {{
            font-size: 13px;
            font-weight: 500;
        }}
        
        .controls select, .controls input {{
            background: #3a3a3a;
            border: 1px solid #4a4a4a;
            color: #e0e0e0;
            padding: 8px 12px;
            border-radius: 5px;
            font-size: 13px;
        }}
        
        .controls button {{
            background: #667eea;
            border: none;
            color: white;
            padding: 8px 20px;
            border-radius: 5px;
            cursor: pointer;
            font-size: 13px;
            font-weight: 500;
            transition: background 0.3s;
        }}
        
        .controls button:hover {{
            background: #5568d3;
        }}
        
        .legend {{
            background: #2a2a2a;
            padding: 15px;
            border-radius: 8px;
            margin-top: 15px;
            font-size: 12px;
        }}
        
        .legend-title {{
            font-weight: bold;
            margin-bottom: 10px;
            font-size: 13px;
        }}
        
        .legend-items {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
            gap: 10px;
        }}
        
        .legend-item {{
            display: flex;
            align-items: center;
            gap: 8px;
        }}
        
        .legend-color {{
            width: 16px;
            height: 16px;
            border-radius: 3px;
        }}
    </style>
</head>
<body>
    <div class="header">
        <h1>🎨 LVMH Client Intelligence Dashboard</h1>
        <div class="stats">
            <div class="stat">
                <span>Nodes:</span>
                <span class="stat-value">{len(nodes)}</span>
            </div>
            <div class="stat">
                <span>Relationships:</span>
                <span class="stat-value">{len(edges)}</span>
            </div>
            <div class="stat">
                <span>Clients:</span>
                <span class="stat-value">{node_types.get('client', 0)}</span>
            </div>
            <div class="stat">
                <span>Segments:</span>
                <span class="stat-value">{node_types.get('segment', 0)}</span>
            </div>
            <div class="stat">
                <span>Concepts:</span>
                <span class="stat-value">{node_types.get('concept', 0) + node_types.get('brand', 0)}</span>
            </div>
        </div>
    </div>
    
    <div class="tabs">
        <div class="tab active" onclick="switchTab('kg')">📊 Knowledge Graph</div>
        <div class="tab" onclick="switchTab('3d')">🌐 3D Embedding Space</div>
    </div>
    
    <div id="kg-content" class="content active">
        <div class="controls">
            <label>Layout:</label>
            <select id="layout-select" onchange="updateLayout()">
                <option value="cose">Force-Directed (COSE)</option>
                <option value="circle">Circle</option>
                <option value="concentric">Concentric</option>
                <option value="grid">Grid</option>
                <option value="breadthfirst">Hierarchical</option>
            </select>
            
            <label>Filter by type:</label>
            <select id="node-filter" onchange="filterNodes()">
                <option value="all">All Nodes</option>
                <option value="bucket">Buckets Only</option>
                <option value="concept">Concepts & Brands</option>
                <option value="client">Clients</option>
                <option value="segment">Segments</option>
                <option value="action">Actions</option>
            </select>
            
            <label>Search:</label>
            <input type="text" id="search-input" placeholder="Search nodes..." 
                   oninput="searchNodes()" />
            
            <button onclick="resetView()">Reset View</button>
            <button onclick="fitToScreen()">Fit to Screen</button>
        </div>
        
        <div id="kg-container"></div>
        
        <div class="legend">
            <div class="legend-title">Node Types</div>
            <div class="legend-items">
                <div class="legend-item">
                    <div class="legend-color" style="background: #f39c12;"></div>
                    <span>Bucket (Category)</span>
                </div>
                <div class="legend-item">
                    <div class="legend-color" style="background: #e74c3c;"></div>
                    <span>Brand/Product</span>
                </div>
                <div class="legend-item">
                    <div class="legend-color" style="background: #3498db;"></div>
                    <span>Concept</span>
                </div>
                <div class="legend-item">
                    <div class="legend-color" style="background: #2ecc71;"></div>
                    <span>Client</span>
                </div>
                <div class="legend-item">
                    <div class="legend-color" style="background: #9b59b6;"></div>
                    <span>Segment</span>
                </div>
                <div class="legend-item">
                    <div class="legend-color" style="background: #e67e22;"></div>
                    <span>Action</span>
                </div>
            </div>
        </div>
    </div>
    
    <div id="3d-content" class="content">
        <iframe id="viz-3d" src="embedding_space_3d.html" frameborder="0"></iframe>
    </div>
    
    <script>
        // Knowledge graph data
        const kgData = {json.dumps(kg_data, ensure_ascii=False)};
        
        let cy;
        let allElements;
        
        // Initialize Cytoscape
        function initCytoscape() {{
            cy = cytoscape({{
                container: document.getElementById('kg-container'),
                elements: kgData.elements,
                
                style: [
                    {{
                        selector: 'node',
                        style: {{
                            'label': 'data(label)',
                            'text-valign': 'center',
                            'text-halign': 'center',
                            'font-size': '10px',
                            'color': '#ffffff',
                            'text-outline-width': 2,
                            'text-outline-color': '#000000',
                            'width': 'label',
                            'height': 'label',
                            'shape': 'ellipse',
                            'padding': '8px',
                            'background-opacity': 0.9
                        }}
                    }},
                    {{
                        selector: 'node[type="bucket"]',
                        style: {{
                            'background-color': '#f39c12',
                            'shape': 'hexagon',
                            'width': 60,
                            'height': 60,
                            'font-size': '14px',
                            'font-weight': 'bold'
                        }}
                    }},
                    {{
                        selector: 'node[type="brand"]',
                        style: {{
                            'background-color': '#e74c3c',
                            'shape': 'diamond',
                            'width': 45,
                            'height': 45,
                            'font-size': '11px'
                        }}
                    }},
                    {{
                        selector: 'node[type="concept"]',
                        style: {{
                            'background-color': '#3498db',
                            'shape': 'ellipse',
                            'width': 35,
                            'height': 35
                        }}
                    }},
                    {{
                        selector: 'node[type="client"]',
                        style: {{
                            'background-color': '#2ecc71',
                            'shape': 'ellipse',
                            'width': 25,
                            'height': 25,
                            'font-size': '8px'
                        }}
                    }},
                    {{
                        selector: 'node[type="segment"]',
                        style: {{
                            'background-color': '#9b59b6',
                            'shape': 'roundrectangle',
                            'width': 50,
                            'height': 50,
                            'font-size': '12px'
                        }}
                    }},
                    {{
                        selector: 'node[type="action"]',
                        style: {{
                            'background-color': '#e67e22',
                            'shape': 'octagon',
                            'width': 40,
                            'height': 40,
                            'font-size': '10px'
                        }}
                    }},
                    {{
                        selector: 'edge',
                        style: {{
                            'width': 1.5,
                            'line-color': '#555',
                            'target-arrow-color': '#555',
                            'target-arrow-shape': 'triangle',
                            'curve-style': 'bezier',
                            'opacity': 0.5
                        }}
                    }},
                    {{
                        selector: ':selected',
                        style: {{
                            'border-width': 3,
                            'border-color': '#fff',
                            'background-opacity': 1
                        }}
                    }}
                ],
                
                layout: {{
                    name: 'cose',
                    animate: true,
                    animationDuration: 1000,
                    nodeDimensionsIncludeLabels: true,
                    idealEdgeLength: 100,
                    nodeRepulsion: 8000,
                    edgeElasticity: 100,
                    gravity: 0.5
                }}
            }});
            
            allElements = cy.elements();
            
            // Add click handler for node info
            cy.on('tap', 'node', function(evt) {{
                const node = evt.target;
                console.log('Node clicked:', node.data());
            }});
        }}
        
        // Layout management
        function updateLayout() {{
            const layout = document.getElementById('layout-select').value;
            cy.layout({{
                name: layout,
                animate: true,
                animationDuration: 1000
            }}).run();
        }}
        
        // Node filtering
        function filterNodes() {{
            const filterType = document.getElementById('node-filter').value;
            
            if (filterType === 'all') {{
                cy.elements().restore();
            }} else {{
                cy.elements().remove();
                
                let nodesToShow = allElements.nodes(`[type="${{filterType}}"]`);
                if (filterType === 'concept') {{
                    nodesToShow = nodesToShow.union(allElements.nodes('[type="brand"]'));
                }}
                
                const connectedEdges = nodesToShow.connectedEdges();
                const connectedNodes = connectedEdges.connectedNodes();
                
                cy.add(nodesToShow);
                cy.add(connectedEdges);
                cy.add(connectedNodes);
                
                updateLayout();
            }}
        }}
        
        // Search functionality
        function searchNodes() {{
            const searchTerm = document.getElementById('search-input').value.toLowerCase();
            
            cy.elements().removeClass('dimmed');
            cy.elements().style('opacity', 1);
            
            if (searchTerm) {{
                const matches = cy.nodes().filter(node => {{
                    const label = node.data('label').toLowerCase();
                    return label.includes(searchTerm);
                }});
                
                if (matches.length > 0) {{
                    cy.elements().not(matches).style('opacity', 0.2);
                    cy.elements().not(matches).addClass('dimmed');
                    
                    // Highlight matches
                    matches.style('opacity', 1);
                    matches.flashClass('highlighted', 1000);
                    
                    // Fit to matches
                    cy.fit(matches, 100);
                }} else {{
                    cy.elements().style('opacity', 1);
                }}
            }}
        }}
        
        // View controls
        function resetView() {{
            cy.elements().restore();
            cy.elements().style('opacity', 1);
            cy.elements().removeClass('dimmed');
            document.getElementById('search-input').value = '';
            document.getElementById('node-filter').value = 'all';
            updateLayout();
        }}
        
        function fitToScreen() {{
            cy.fit(undefined, 50);
        }}
        
        // Tab switching
        function switchTab(tab) {{
            // Update tab buttons
            document.querySelectorAll('.tab').forEach(t => t.classList.remove('active'));
            event.target.classList.add('active');
            
            // Update content
            document.querySelectorAll('.content').forEach(c => c.classList.remove('active'));
            document.getElementById(tab + '-content').classList.add('active');
        }}
        
        // Initialize on load
        window.addEventListener('DOMContentLoaded', () => {{
            initCytoscape();
        }});
    </script>
</body>
</html>
"""
    
    # Write dashboard
    output_path = DEMO_DIR / "dashboard.html"
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(html_content)
    
    log_stage("dashboard", f"Dashboard generated: {output_path}")
    return output_path


if __name__ == "__main__":
    generate_dashboard()
