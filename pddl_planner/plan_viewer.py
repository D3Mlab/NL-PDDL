#!/usr/bin/env python3
"""
Regression Plan Tree Viewer for NL-PDDL.

Parses regression planning result files and generates an interactive HTML
dashboard that visualizes the goal-regression search tree using D3.js.

Usage:
    python -m pddl_planner.plan_viewer <file_or_directory> [-o output.html]
"""

import argparse
import json
import os
import re
import sys
import webbrowser
from pathlib import Path
from typing import Dict, List, Optional, Tuple


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------

def split_action_list(action_list_str: str) -> List[str]:
    """Split an action list string like "[Action(...), Action(...)]" into
    individual action strings by tracking balanced parentheses."""
    s = action_list_str.strip()
    if s == "[]":
        return []
    if not s.startswith("[") or not s.endswith("]"):
        return []
    s = s[1:-1].strip()
    actions = []
    depth = 0
    start = 0
    for i, ch in enumerate(s):
        if ch == "(":
            depth += 1
        elif ch == ")":
            depth -= 1
            if depth == 0:
                actions.append(s[start : i + 1].strip())
                start = i + 1
                while start < len(s) and s[start] in ", ":
                    start += 1
    return actions


def parse_result_file(filepath: str) -> dict:
    """Parse a regression result .txt file into structured data.

    Returns a dict with keys:
        - "filepath": str
        - "filename": str
        - "initial_state": {"state": str, "actions": str, "bindings": str}
        - "subgoals": list of dicts with keys:
              "id" (int), "label" (str), "state" (str),
              "actions_str" (str), "actions" (list[str]),
              "bindings" (str), "depth" (int)
    """
    with open(filepath, "r", encoding="utf-8") as f:
        content = f.read()

    lines = content.split("\n")
    blocks: List[dict] = []
    current_header: Optional[str] = None
    current_lines: List[str] = []

    def flush():
        nonlocal current_header, current_lines
        if current_header is not None:
            blocks.append({"header": current_header, "lines": current_lines})
        current_header = None
        current_lines = []

    i = 0
    while i < len(lines):
        line = lines[i]
        stripped = line.strip()
        if stripped == "--------------------":
            flush()
            i += 1
            continue
        if stripped == "Regressed goals:":
            i += 1
            continue
        if stripped.startswith("Initial State:") or stripped.startswith("Subgoal S"):
            flush()
            current_header = stripped
            i += 1
            continue
        if current_header is not None:
            current_lines.append(line)
        i += 1
    flush()

    initial_state = None
    subgoals = []

    for block in blocks:
        header = block["header"]
        blines = block["lines"]
        while blines and not blines[-1].strip():
            blines.pop()
        state_str = blines[0].strip() if len(blines) > 0 else ""
        actions_str = blines[1].strip() if len(blines) > 1 else "[]"
        bindings_str = blines[2].strip() if len(blines) > 2 else "{}"

        if header.startswith("Initial State"):
            initial_state = {
                "state": state_str,
                "actions": actions_str,
                "bindings": bindings_str,
            }
        elif header.startswith("Subgoal S"):
            m = re.match(r"Subgoal S(\d+):", header)
            idx = int(m.group(1)) if m else len(subgoals)
            actions = split_action_list(actions_str)
            subgoals.append({
                "id": idx,
                "label": f"S{idx}",
                "state": state_str,
                "actions_str": actions_str,
                "actions": actions,
                "bindings": bindings_str,
                "depth": len(actions),
            })

    return {
        "filepath": filepath,
        "filename": os.path.basename(filepath),
        "initial_state": initial_state,
        "subgoals": subgoals,
    }


# ---------------------------------------------------------------------------
# Tree building
# ---------------------------------------------------------------------------

def _action_list_key(actions: List[str]) -> str:
    """Create a hashable key from a list of action strings."""
    return "|||".join(actions)


def build_tree(parsed: dict) -> dict:
    """Build a nested tree structure from parsed subgoals.

    The parent of subgoal with actions [a1, a2, ..., an] is the first
    subgoal whose actions equal [a2, ..., an]. S0 (empty actions) is root.

    Returns a nested dict suitable for D3 hierarchical layouts.
    """
    subgoals = parsed["subgoals"]
    if not subgoals:
        return {}

    key_to_indices: Dict[str, List[int]] = {}
    for i, sg in enumerate(subgoals):
        k = _action_list_key(sg["actions"])
        key_to_indices.setdefault(k, []).append(i)

    nodes = []
    for sg in subgoals:
        node = {
            "id": sg["id"],
            "label": sg["label"],
            "state": sg["state"],
            "actions": sg["actions"],
            "actions_str": sg["actions_str"],
            "bindings": sg["bindings"],
            "depth": sg["depth"],
            "edge_action": sg["actions"][0] if sg["actions"] else None,
            "children": [],
        }
        nodes.append(node)

    # For each non-root node, find its parent by suffix matching
    used_counts: Dict[str, int] = {}
    for i, sg in enumerate(subgoals):
        if not sg["actions"]:
            continue
        parent_key = _action_list_key(sg["actions"][1:])
        candidates = key_to_indices.get(parent_key, [])
        # Pick the first candidate with index < i (BFS order guarantees parent is earlier)
        parent_idx = None
        for c in candidates:
            if c < i:
                parent_idx = c
                break
        if parent_idx is not None:
            nodes[parent_idx]["children"].append(nodes[i])

    if not nodes:
        return {"id": -1, "label": "(empty)", "state": "", "actions": [],
                "actions_str": "[]", "bindings": "{}", "depth": 0,
                "edge_action": None, "children": []}
    return nodes[0]


# ---------------------------------------------------------------------------
# HTML generation
# ---------------------------------------------------------------------------

HTML_TEMPLATE = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>NL-PDDL Regression Plan Viewer</title>
<script src="https://d3js.org/d3.v7.min.js"></script>
<style>
:root {
    --bg: #0f1117;
    --panel-bg: #1a1d27;
    --border: #2a2d3a;
    --text: #e0e0e8;
    --text-dim: #8888a0;
    --accent: #6c8cff;
    --accent2: #ff6c6c;
    --highlight: #ffcc44;
    --link-color: #3a3f55;
}
* { margin: 0; padding: 0; box-sizing: border-box; }
body {
    font-family: 'Segoe UI', system-ui, -apple-system, sans-serif;
    background: var(--bg);
    color: var(--text);
    overflow: hidden;
    height: 100vh;
}
#app {
    display: grid;
    grid-template-rows: auto auto 1fr;
    grid-template-columns: 1fr 400px;
    height: 100vh;
}
#topbar {
    grid-column: 1 / -1;
    display: flex;
    align-items: center;
    gap: 14px;
    padding: 8px 20px;
    background: var(--panel-bg);
    border-bottom: 1px solid var(--border);
    flex-wrap: wrap;
}
#topbar h1 { font-size: 15px; font-weight: 600; color: var(--accent); white-space: nowrap; }
#topbar select, #topbar button {
    background: var(--bg); color: var(--text);
    border: 1px solid var(--border); border-radius: 6px;
    padding: 4px 10px; font-size: 12px; cursor: pointer;
}
#topbar button:hover { border-color: var(--accent); }
#topbar button.active { background: var(--accent); color: #fff; border-color: var(--accent); }
.slider-group {
    display: flex; align-items: center; gap: 8px; flex: 1; min-width: 200px;
}
.slider-group label { font-size: 12px; color: var(--text-dim); white-space: nowrap; }
.slider-group input[type=range] { flex: 1; accent-color: var(--accent); }
.slider-group .val { font-size: 13px; font-weight: 600; min-width: 50px; }
.stats { font-size: 11px; color: var(--text-dim); white-space: nowrap; }
.btn-group { display: flex; gap: 4px; }
#tree-container { grid-row: 3; grid-column: 1; position: relative; overflow: hidden; }
#tree-svg { width: 100%; height: 100%; }
.node circle {
    cursor: pointer; stroke-width: 2px;
    transition: fill 0.2s, stroke 0.2s, r 0.2s;
}
.node text { font-size: 11px; fill: var(--text); pointer-events: none; }
.link { fill: none; stroke: var(--link-color); stroke-width: 1.5px; transition: stroke 0.2s, stroke-width 0.2s; }
.link.highlighted { stroke: var(--highlight); stroke-width: 3px; }
.node.highlighted circle { stroke: var(--highlight) !important; fill: var(--highlight) !important; }
.node.selected circle { stroke: var(--accent2) !important; stroke-width: 3.5px; }
.edge-label-g rect {
    fill: #1e2030; stroke: #333a50; stroke-width: 1px; rx: 4; ry: 4;
}
.edge-label-g text {
    font-size: 9px; fill: #9898b8; pointer-events: none;
    font-family: 'JetBrains Mono', 'Fira Code', 'Menlo', monospace;
}
.edge-label-g { pointer-events: none; }
#detail-panel {
    grid-row: 3; grid-column: 2;
    background: var(--panel-bg); border-left: 1px solid var(--border);
    overflow-y: auto; padding: 16px 18px;
}
#detail-panel h2 {
    font-size: 14px; color: var(--accent);
    margin-bottom: 10px; padding-bottom: 8px; border-bottom: 1px solid var(--border);
}
#detail-panel h3 {
    font-size: 11px; color: var(--text-dim); text-transform: uppercase;
    letter-spacing: 0.5px; margin: 14px 0 6px;
}
#detail-panel .cb {
    background: var(--bg); border: 1px solid var(--border); border-radius: 6px;
    padding: 10px 12px; font-size: 11.5px; line-height: 1.7;
    word-break: break-word; white-space: pre-wrap;
    font-family: 'JetBrains Mono', 'Fira Code', 'Menlo', monospace;
    max-height: 220px; overflow-y: auto;
}
.path-action {
    display: flex; align-items: flex-start; gap: 8px; margin: 3px 0;
}
.step-num {
    background: var(--accent); color: #fff; border-radius: 50%;
    min-width: 20px; height: 20px; display: flex; align-items: center;
    justify-content: center; font-size: 10px; font-weight: 700; flex-shrink: 0;
}
.step-text {
    font-family: 'JetBrains Mono', 'Fira Code', 'Menlo', monospace;
    font-size: 11px; line-height: 1.5;
}
.empty-state { color: var(--text-dim); font-size: 13px; text-align: center; margin-top: 60px; }
.tooltip {
    position: absolute; background: var(--panel-bg); border: 1px solid var(--border);
    border-radius: 6px; padding: 8px 12px; font-size: 11px;
    pointer-events: none; opacity: 0; transition: opacity 0.15s;
    max-width: 380px; z-index: 100; box-shadow: 0 4px 20px rgba(0,0,0,0.5);
    line-height: 1.5;
}
.tooltip.visible { opacity: 1; }
.initial-state-bar {
    grid-column: 1 / -1;
    grid-row: 2;
    padding: 4px 20px;
    background: #141620;
    border-bottom: 1px solid var(--border);
    font-size: 11px;
    color: var(--text-dim);
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
    max-height: 28px;
    line-height: 20px;
}
.initial-state-bar strong { color: var(--accent); margin-right: 6px; }
</style>
</head>
<body>
<div id="app">
    <div id="topbar">
        <h1>NL-PDDL Regression Viewer</h1>
        <div class="slider-group">
            <label>Iteration:</label>
            <input type="range" id="iter-slider" min="0" max="0" value="0">
            <span class="val" id="iter-val">S0</span>
            <button id="play-btn" title="Auto-play">&#9654;</button>
            <button id="reset-btn" title="Show All">&#8634;</button>
        </div>
        <div class="btn-group">
            <button id="expand-btn" title="Expand All">+ All</button>
            <button id="collapse-btn" title="Collapse to depth 2">&minus; All</button>
            <button id="fit-btn" title="Fit to screen">Fit</button>
        </div>
        <div class="stats" id="stats-info"></div>
        <div class="btn-group" id="file-nav" style="display:none;">
            <button id="prev-file" title="Previous file">&larr;</button>
            <span id="file-label" style="font-size:11px;color:var(--text-dim);padding:0 6px;display:flex;align-items:center;"></span>
            <button id="next-file" title="Next file">&rarr;</button>
        </div>
    </div>
    <div id="init-bar" class="initial-state-bar" style="display:none;"></div>
    <div id="tree-container">
        <svg id="tree-svg"></svg>
        <div class="tooltip" id="tooltip"></div>
    </div>
    <div id="detail-panel">
        <div class="empty-state">Click a node to view details<br><br>
        <span style="font-size:11px">Double-click to expand/collapse subtrees</span></div>
    </div>
</div>

<script>
const ALL_DATA = __DATA_PLACEHOLDER__;

const DEPTH_COLORS = [
    '#6c8cff','#8c6cff','#ff6cb0','#ff8c6c','#6cddff',
    '#44cc88','#ddaa44','#cc66cc','#66cccc','#aabb44',
    '#ff6666','#6699ff','#cc8844','#88cc66','#dd88cc'
];
function colD(d) { return DEPTH_COLORS[d % DEPTH_COLORS.length]; }
function parseActionName(a) { const m = a && a.match(/Action\('([^']+)'/); return m ? m[1] : (a||''); }
function parseActionFull(a) {
    const m = a && a.match(/Action\('([^']+)'\((\[[^\]]*\])\)\)/);
    return m ? { name: m[1], params: m[2] } : { name: a||'', params: '' };
}
function groundAction(a) {
    const m = a && a.match(/Action\('([^']+)'\((\[[^\]]*\])\)\)/);
    if (!m) return a || '';
    const name = m[1];
    const paramStr = m[2].slice(1, -1).trim();
    const params = paramStr ? paramStr.split(/,\s*/) : [];
    const seen = new Set();
    const vars = [];
    const varRe = /\?[^\s,\])]+/g;
    let vm;
    while ((vm = varRe.exec(name)) !== null) {
        if (!seen.has(vm[0])) { seen.add(vm[0]); vars.push(vm[0]); }
    }
    let result = name;
    vars.forEach((v, i) => {
        if (i < params.length) {
            result = result.split(v).join(params[i]);
        }
    });
    return result;
}

let currentDataIdx = 0;
let root = null;
let g, zoomBehavior;
let selectedNode = null;
let playing = false, playTimer = null;
let maxIter = 0;

const svgEl = d3.select('#tree-svg');
const containerEl = document.getElementById('tree-container');
const slider = document.getElementById('iter-slider');
const iterVal = document.getElementById('iter-val');
const playBtn = document.getElementById('play-btn');
const resetBtn = document.getElementById('reset-btn');
const expandBtn = document.getElementById('expand-btn');
const collapseBtn = document.getElementById('collapse-btn');
const fitBtn = document.getElementById('fit-btn');
const fileNav = document.getElementById('file-nav');
const fileLabel = document.getElementById('file-label');
const prevFileBtn = document.getElementById('prev-file');
const nextFileBtn = document.getElementById('next-file');

if (ALL_DATA.length > 1) {
    fileNav.style.display = 'flex';
    prevFileBtn.addEventListener('click', () => { if (currentDataIdx > 0) loadTree(currentDataIdx - 1); });
    nextFileBtn.addEventListener('click', () => { if (currentDataIdx < ALL_DATA.length - 1) loadTree(currentDataIdx + 1); });
}

slider.addEventListener('input', () => { maxIter = +slider.value; iterVal.textContent = `S${maxIter}`; updateTree(root); });
playBtn.addEventListener('click', () => { playing ? stopPlay() : startPlay(); });
resetBtn.addEventListener('click', () => { stopPlay(); slider.value = slider.max; maxIter = +slider.max; iterVal.textContent = `S${maxIter}`; expandAll(root); updateTree(root); });
expandBtn.addEventListener('click', () => { expandAll(root); updateTree(root); });
collapseBtn.addEventListener('click', () => { collapseToDepth(root, 2); updateTree(root); });
fitBtn.addEventListener('click', fitView);

function startPlay() {
    playing = true; playBtn.classList.add('active'); playBtn.innerHTML = '&#9646;&#9646;';
    maxIter = 0; slider.value = 0; iterVal.textContent = 'S0';
    expandAll(root); updateTree(root);
    playTimer = setInterval(() => {
        maxIter++;
        if (maxIter > +slider.max) { stopPlay(); return; }
        slider.value = maxIter; iterVal.textContent = `S${maxIter}`;
        updateTree(root);
    }, 300);
}
function stopPlay() {
    playing = false; playBtn.classList.remove('active'); playBtn.innerHTML = '&#9654;';
    if (playTimer) { clearInterval(playTimer); playTimer = null; }
}

function collapse(d) {
    if (d.children) { d._children = d.children; d.children = null; }
}
function expand(d) {
    if (d._children) { d.children = d._children; d._children = null; }
}
function expandAll(d) {
    if (!d) return;
    expand(d);
    if (d.children) d.children.forEach(expandAll);
    if (d._children) d._children.forEach(expandAll);
}
function collapseToDepth(d, maxD) {
    if (!d) return;
    expand(d);
    if (d.depth >= maxD) { collapse(d); return; }
    if (d.children) d.children.forEach(c => collapseToDepth(c, maxD));
}
function allChildren(d) { return d.children || d._children || []; }

function loadTree(idx) {
    stopPlay(); selectedNode = null; currentDataIdx = idx;
    const data = ALL_DATA[idx];
    document.getElementById('stats-info').textContent = `${data.subgoal_count} subgoals | max depth ${data.max_depth}`;
    if (ALL_DATA.length > 1) {
        fileLabel.textContent = `${idx+1}/${ALL_DATA.length}: ${data.filename}`;
        prevFileBtn.disabled = idx === 0;
        nextFileBtn.disabled = idx === ALL_DATA.length - 1;
    }
    const initBar = document.getElementById('init-bar');
    if (data.initial_state) {
        initBar.innerHTML = `<strong>Initial State:</strong> ${data.initial_state.state}`;
        initBar.style.display = '';
    } else { initBar.style.display = 'none'; }

    slider.max = Math.max(0, data.subgoal_count - 1);
    slider.value = slider.max;
    maxIter = +slider.max;
    iterVal.textContent = data.subgoal_count > 0 ? `S${maxIter}` : '(empty)';

    svgEl.selectAll('*').remove();
    const w = containerEl.clientWidth, h = containerEl.clientHeight;
    svgEl.attr('width', w).attr('height', h);
    zoomBehavior = d3.zoom().scaleExtent([0.02, 6]).on('zoom', e => g.attr('transform', e.transform));
    svgEl.call(zoomBehavior);
    g = svgEl.append('g');
    g.append('g').attr('class', 'links-g');
    g.append('g').attr('class', 'labels-g');
    g.append('g').attr('class', 'nodes-g');

    root = d3.hierarchy(data.tree, d => d.children);
    root.x0 = h / 2; root.y0 = 0;

    if (data.subgoal_count > 80) collapseToDepth(root, 2);

    updateTree(root);
    showDetailPanel(null);
    setTimeout(fitView, 100);
}

function updateTree(source) {
    if (!root || !g) return;
    const w = containerEl.clientWidth, h = containerEl.clientHeight;
    const dur = 350;

    const visibleNodes = root.descendants();
    const leafCount = visibleNodes.filter(d => !(d.children && d.children.length) && !(d._children && d._children.length)).length || 1;
    const dynamicH = Math.max(h - 40, leafCount * 55);
    const maxDepth = d3.max(visibleNodes, d => d.depth) || 1;
    const totalY = maxDepth * 260;
    const treeLayout = d3.tree().size([dynamicH, totalY]).separation((a, b) => a.parent === b.parent ? 1 : 1.6);
    treeLayout(root);
    // Reverse horizontal axis so time flows left-to-right:
    // deeper subgoals on the left, final goal (root) on the right.
    root.descendants().forEach(d => { d.y = totalY - d.y; });

    // --- Links ---
    const linkData = root.links().filter(l => l.target.data.id <= maxIter);
    const linkSel = g.select('.links-g').selectAll('.link').data(linkData, d => d.target.data.id);
    const linkEnter = linkSel.enter().append('path').attr('class', 'link')
        .attr('d', () => { const o = {x: source.x0||0, y: source.y0||0}; return diagLine(o, o); })
        .attr('opacity', 0);
    linkEnter.merge(linkSel).transition().duration(dur)
        .attr('d', d => diagLine(d.source, d.target)).attr('opacity', 1);
    linkSel.exit().transition().duration(dur)
        .attr('d', () => { const o = {x: source.x||0, y: source.y||0}; return diagLine(o, o); })
        .attr('opacity', 0).remove();

    // --- Nodes ---
    const nodeData = root.descendants().filter(d => d.data.id <= maxIter);
    const nodeSel = g.select('.nodes-g').selectAll('.node').data(nodeData, d => d.data.id);
    const nodeEnter = nodeSel.enter().append('g').attr('class', 'node')
        .attr('transform', () => `translate(${source.y0||0},${source.x0||0})`)
        .attr('opacity', 0);
    nodeEnter.append('circle');
    nodeEnter.append('text').attr('dy', -12).attr('text-anchor', 'middle');

    const nodeMerge = nodeEnter.merge(nodeSel);
    nodeMerge.transition().duration(dur)
        .attr('transform', d => `translate(${d.y},${d.x})`).attr('opacity', 1);
    nodeMerge.select('circle')
        .attr('r', d => {
            const hasKids = (d.children && d.children.length) || (d._children && d._children.length);
            return d.data.id === 0 ? 9 : (hasKids ? 6.5 : 5);
        })
        .attr('fill', d => {
            if (d._children && d._children.length) return '#1a1d27';
            return colD(d.data.depth);
        })
        .attr('stroke', d => colD(d.data.depth))
        .attr('stroke-width', d => d._children && d._children.length ? 2.5 : 2);
    nodeMerge.select('text').text(d => {
        const lbl = d.data.id === 0 ? 'Goal' : d.data.label;
        return `${lbl} (d${d.data.depth})`;
    });

    nodeMerge
        .on('click', (e, d) => { e.stopPropagation(); onNodeClick(d); })
        .on('dblclick', (e, d) => { e.stopPropagation(); toggleNode(d); })
        .on('mouseenter', (e, d) => showTooltip(e, d))
        .on('mouseleave', hideTooltip);

    nodeSel.exit().transition().duration(dur)
        .attr('transform', () => `translate(${source.y||0},${source.x||0})`)
        .attr('opacity', 0).remove();

    // --- Edge labels (boxed, multi-line, positioned near target) ---
    const MAX_LABEL_WIDTH = 20;
    function wrapText(str) {
        const words = str.split(/\s+/);
        const lines = [];
        let cur = '';
        words.forEach(w => {
            if (cur && (cur + ' ' + w).length > MAX_LABEL_WIDTH) { lines.push(cur); cur = w; }
            else { cur = cur ? cur + ' ' + w : w; }
        });
        if (cur) lines.push(cur);
        return lines;
    }

    const edgeData = root.links().filter(l => l.target.data.id <= maxIter && l.target.data.edge_action);
    const edgeSel = g.select('.labels-g').selectAll('.edge-label-g').data(edgeData, d => d.target.data.id);
    const edgeEnter = edgeSel.enter().append('g').attr('class', 'edge-label-g').attr('opacity', 0);
    edgeEnter.append('rect');
    edgeEnter.append('text').attr('text-anchor', 'middle');

    const edgeMerge = edgeEnter.merge(edgeSel);
    edgeMerge.each(function(d) {
        const txt = groundAction(d.target.data.edge_action);
        const lines = wrapText(txt);
        const textEl = d3.select(this).select('text');
        textEl.selectAll('tspan').remove();
        const lineH = 12;
        const topOffset = -(lines.length - 1) * lineH / 2;
        lines.forEach((ln, i) => {
            textEl.append('tspan')
                .attr('x', 0)
                .attr('dy', i === 0 ? topOffset : lineH)
                .text(ln);
        });
        const bbox = textEl.node().getBBox();
        const padX = 7, padY = 4;
        d3.select(this).select('rect')
            .attr('x', bbox.x - padX)
            .attr('y', bbox.y - padY)
            .attr('width', bbox.width + padX * 2)
            .attr('height', bbox.height + padY * 2)
            .attr('rx', 5).attr('ry', 5);
    });
    edgeMerge.transition().duration(dur)
        .attr('transform', d => {
            const t = 0.7;
            const mx = d.source.y * (1 - t) + d.target.y * t;
            const my = d.source.x * (1 - t) + d.target.x * t;
            return `translate(${mx},${my})`;
        })
        .attr('opacity', 0.9);
    edgeSel.exit().transition().duration(dur).attr('opacity', 0).remove();

    // Save positions for transitions
    root.descendants().forEach(d => { d.x0 = d.x; d.y0 = d.y; });

    if (selectedNode) reHighlight();
}

function diagLine(s, d) {
    return `M${s.y},${s.x}C${(s.y + d.y) / 2},${s.x} ${(s.y + d.y) / 2},${d.x} ${d.y},${d.x}`;
}

function toggleNode(d) {
    if (d.children) { d._children = d.children; d.children = null; }
    else if (d._children) { d.children = d._children; d._children = null; }
    updateTree(d);
}

function onNodeClick(d) {
    selectedNode = d;
    reHighlight();
    showDetailPanel(d);
}

function reHighlight() {
    if (!selectedNode || !g) return;
    const pathIds = new Set();
    const pathLinkIds = new Set();
    let cur = selectedNode;
    while (cur) { pathIds.add(cur.data.id); if (cur.parent) pathLinkIds.add(cur.data.id); cur = cur.parent; }
    g.selectAll('.node')
        .classed('highlighted', d => pathIds.has(d.data.id) && d !== selectedNode)
        .classed('selected', d => d === selectedNode);
    g.selectAll('.link')
        .classed('highlighted', d => pathLinkIds.has(d.target.data.id));
}

function fitView() {
    if (!g || !g.node()) return;
    const w = containerEl.clientWidth, h = containerEl.clientHeight;
    const bounds = g.node().getBBox();
    if (!bounds.width || !bounds.height) return;
    const scale = Math.min(w / (bounds.width + 100), h / (bounds.height + 60), 1.5);
    const tx = (w - bounds.width * scale) / 2 - bounds.x * scale;
    const ty = (h - bounds.height * scale) / 2 - bounds.y * scale;
    svgEl.transition().duration(500).call(zoomBehavior.transform, d3.zoomIdentity.translate(tx, ty).scale(scale));
}

function showTooltip(event, d) {
    const tip = document.getElementById('tooltip');
    const acts = (d.data.actions || []).map(groundAction);
    const hasKids = (d.children && d.children.length) || (d._children && d._children.length);
    const kidCount = (d.children ? d.children.length : 0) + (d._children ? d._children.length : 0);
    const tipLabel = d.data.id === 0 ? 'Goal (S0)' : d.data.label;
    let html = `<strong>${tipLabel}</strong> &nbsp;depth ${d.data.depth}`;
    if (kidCount) html += ` &nbsp;|&nbsp; ${kidCount} children${d._children ? ' (collapsed)' : ''}`;
    html += `<br>`;
    if (acts.length) html += `<span style="color:var(--accent)">${acts.join(' &rarr; ')}</span><br>`;
    else html += `<span style="color:var(--text-dim)">Goal (root)</span><br>`;
    const st = d.data.state.replace(/^\(\(/, '').replace(/\)\)$/, '');
    html += `<span style="color:var(--text-dim)">${st.length > 150 ? st.substring(0,147)+'...' : st}</span>`;
    tip.innerHTML = html;
    const rect = containerEl.getBoundingClientRect();
    let left = event.clientX - rect.left + 15;
    let top = event.clientY - rect.top + 15;
    if (left + 350 > rect.width) left = rect.width - 360;
    if (top + 100 > rect.height) top = event.clientY - rect.top - 80;
    tip.style.left = left + 'px'; tip.style.top = top + 'px';
    tip.classList.add('visible');
}
function hideTooltip() { document.getElementById('tooltip').classList.remove('visible'); }

function showDetailPanel(d) {
    const panel = document.getElementById('detail-panel');
    if (!d) {
        panel.innerHTML = '<div class="empty-state">Click a node to view details<br><br><span style="font-size:11px">Double-click to expand/collapse subtrees</span></div>';
        return;
    }
    const data = d.data;
    // Walking from selected node to root gives forward execution order
    // (first-to-execute at index 0, goal-achieving action at end)
    let pathActions = [];
    let cur = d;
    while (cur && cur.parent) { pathActions.push(cur.data.edge_action); cur = cur.parent; }

    const preds = data.state.replace(/^\(\(/, '').replace(/\)\)$/, '').split(' \u2227 ').map(s=>s.trim()).filter(Boolean);
    const displayLabel = data.id === 0 ? 'Goal (S0)' : data.label;

    let h = `<h2>${displayLabel} <span style="color:var(--text-dim);font-size:12px">depth ${data.depth}</span></h2>`;

    h += '<h3>Execution Plan (preconditions &rarr; goal)</h3>';
    if (!pathActions.length) {
        h += '<div class="cb" style="color:var(--text-dim)">This is the goal node &mdash; no actions needed</div>';
    } else {
        h += '<div style="margin:4px 0">';
        pathActions.forEach((a, i) => {
            const grounded = groundAction(a);
            h += `<div class="path-action"><span class="step-num">${i+1}</span><span class="step-text">${grounded}</span></div>`;
        });
        h += `<div class="path-action" style="margin-top:4px"><span class="step-num" style="background:var(--highlight);color:#000">&#10003;</span><span class="step-text" style="color:var(--highlight)">Goal achieved</span></div>`;
        h += '</div>';
    }

    h += '<h3>Required State (Preconditions)</h3><div class="cb">';
    preds.forEach(p => { h += p + '\n'; });
    h += '</div>';

    if (data.actions && data.actions.length) {
        h += '<h3>Action Sequence (as in file)</h3><div class="cb">';
        data.actions.forEach((a, i) => {
            const p = parseActionFull(a);
            h += `${i+1}. ${p.name} ${p.params}\n`;
        });
        h += '</div>';
        h += '<h3>Grounded Actions</h3><div class="cb">';
        data.actions.forEach((a, i) => {
            h += `${i+1}. ${groundAction(a)}\n`;
        });
        h += '</div>';
    }

    const kidCount = (d.children ? d.children.length : 0) + (d._children ? d._children.length : 0);
    if (kidCount) {
        h += `<h3>Children (${kidCount})</h3><div class="cb" style="max-height:120px">`;
        allChildren(d).forEach(c => {
            h += `${c.data.label}: ${groundAction(c.data.edge_action)}\n`;
        });
        h += '</div>';
    }

    panel.innerHTML = h;
}

svgEl.on('click', () => { selectedNode = null; g.selectAll('.node').classed('highlighted',false).classed('selected',false); g.selectAll('.link').classed('highlighted',false); showDetailPanel(null); });

if (ALL_DATA.length > 0) loadTree(0);
window.addEventListener('resize', () => { if (root) { const w=containerEl.clientWidth,h=containerEl.clientHeight; svgEl.attr('width',w).attr('height',h); updateTree(root); }});
</script>
</body>
</html>"""


def generate_html(all_parsed: List[dict], output_path: str) -> str:
    """Generate a self-contained HTML file with embedded tree data."""
    data_for_js = []
    for parsed in all_parsed:
        tree = build_tree(parsed)
        max_depth = max((sg["depth"] for sg in parsed["subgoals"]), default=0)
        data_for_js.append({
            "filename": parsed["filename"],
            "filepath": parsed["filepath"],
            "initial_state": parsed["initial_state"],
            "tree": tree,
            "subgoal_count": len(parsed["subgoals"]),
            "max_depth": max_depth,
        })

    json_str = json.dumps(data_for_js, ensure_ascii=False)
    html = HTML_TEMPLATE.replace("__DATA_PLACEHOLDER__", json_str)

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html)

    return output_path


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def discover_result_files(path: str) -> List[str]:
    """Find all result .txt files in a path (file or directory)."""
    p = Path(path)
    if p.is_file():
        return [str(p)]
    if p.is_dir():
        files = sorted(p.glob("*.txt"))
        return [str(f) for f in files if f.name != "llm_calls.txt"]
    return []


def main():
    parser = argparse.ArgumentParser(
        description="NL-PDDL Regression Plan Tree Viewer"
    )
    parser.add_argument(
        "path",
        help="Path to a result .txt file or a directory containing them",
    )
    parser.add_argument(
        "-o", "--output",
        default=None,
        help="Output HTML file path (default: plan_viewer_output.html in the same directory)",
    )
    parser.add_argument(
        "--no-open",
        action="store_true",
        help="Do not automatically open the HTML in a browser",
    )
    args = parser.parse_args()

    files = discover_result_files(args.path)
    if not files:
        print(f"Error: no result files found at '{args.path}'", file=sys.stderr)
        sys.exit(1)

    print(f"Found {len(files)} result file(s):")
    for f in files:
        print(f"  {f}")

    all_parsed = []
    for f in files:
        parsed = parse_result_file(f)
        all_parsed.append(parsed)
        print(f"  Parsed {parsed['filename']}: {len(parsed['subgoals'])} subgoals")

    if args.output:
        output_path = args.output
    else:
        base = Path(args.path)
        if base.is_dir():
            output_path = str(base / "plan_viewer_output.html")
        else:
            output_path = str(base.parent / "plan_viewer_output.html")

    generate_html(all_parsed, output_path)
    print(f"\nGenerated: {output_path}")

    if not args.no_open:
        webbrowser.open(f"file://{os.path.abspath(output_path)}")


if __name__ == "__main__":
    main()
