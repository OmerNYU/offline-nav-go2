"""Semantic memory store for topological navigation."""

from typing import Optional

from memory.types import MemoryNode, Pose2D


class SemanticMemoryStore:
    """In-memory store for semantic navigation nodes.
    
    Maintains a dictionary of memory nodes indexed by node_id.
    """
    
    def __init__(self) -> None:
        """Initialize empty memory store."""
        self.nodes: dict[int, MemoryNode] = {}
        self._next_id: int = 0
    
    def add_node(
        self,
        pose: Pose2D,
        embedding: Optional[list[float]],
        tags: list[str],
        summary: str
    ) -> int:
        """Add a new memory node to the store.
        
        Args:
            pose: 2D pose of the node.
            embedding: Optional pre-computed embedding vector.
            tags: List of semantic tags.
            summary: Natural language description.
            
        Returns:
            node_id: Unique identifier for the created node.
        """
        node_id = self._next_id
        self._next_id += 1
        
        node = MemoryNode(
            node_id=node_id,
            pose=pose,
            embedding=embedding,
            tags=tags,
            summary=summary
        )
        self.nodes[node_id] = node
        return node_id
    
    def get_node(self, node_id: int) -> Optional[MemoryNode]:
        """Retrieve a node by its ID.
        
        Args:
            node_id: Node identifier.
            
        Returns:
            MemoryNode if found, None otherwise.
        """
        return self.nodes.get(node_id)
    
    def all_nodes(self) -> list[MemoryNode]:
        """Get all nodes in the store.
        
        Returns:
            List of all memory nodes.
        """
        return list(self.nodes.values())


def seed_demo_store() -> SemanticMemoryStore:
    """Create a demo store with sample nodes for testing.
    
    Returns:
        SemanticMemoryStore with 5 demo nodes.
    """
    store = SemanticMemoryStore()
    
    # Node 0: Kitchen area
    store.add_node(
        pose=Pose2D(x=1.0, y=2.0, yaw=0.0),
        embedding=None,
        tags=["kitchen", "appliances", "counter"],
        summary="Kitchen area with counter and appliances"
    )
    
    # Node 1: Living room
    store.add_node(
        pose=Pose2D(x=5.0, y=2.0, yaw=1.57),
        embedding=None,
        tags=["living room", "couch", "table"],
        summary="Living room with couch and coffee table"
    )
    
    # Node 2: Hallway
    store.add_node(
        pose=Pose2D(x=3.0, y=0.0, yaw=0.0),
        embedding=None,
        tags=["hallway", "corridor", "doorway"],
        summary="Hallway connecting rooms with doorways"
    )
    
    # Node 3: Bedroom
    store.add_node(
        pose=Pose2D(x=8.0, y=5.0, yaw=-1.57),
        embedding=None,
        tags=["bedroom", "bed", "closet"],
        summary="Bedroom with bed and closet"
    )
    
    # Node 4: Bathroom
    store.add_node(
        pose=Pose2D(x=2.0, y=5.0, yaw=3.14),
        embedding=None,
        tags=["bathroom", "sink", "shower"],
        summary="Bathroom with sink and shower"
    )
    
    return store

