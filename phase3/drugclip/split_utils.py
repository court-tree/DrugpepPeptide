"""Leakage graph construction and connected-component split."""

from __future__ import annotations

from collections import deque

def connected_components(edges: dict[str, set[str]]) -> list[set[str]]:
    seen: set[str] = set()
    components: list[set[str]] = []
    for node in sorted(edges):
        if node in seen:
            continue
        component: set[str] = set()
        queue = deque([node])
        seen.add(node)
        while queue:
            current = queue.popleft()
            component.add(current)
            for nxt in edges[current]:
                if nxt not in seen:
                    seen.add(nxt)
                    queue.append(nxt)
        components.append(component)
    return components
