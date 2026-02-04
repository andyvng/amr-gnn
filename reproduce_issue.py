import torch
import torch.nn as nn
from captum.attr import IntegratedGradients


# Mock GNNModel to simulate the issue
class MockGNNModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(10, 1)

    def forward(self, x, edge_index):
        # Check if edge_index was expanded
        print(f"Forward called with x shape: {x.shape}")
        print(f"Forward called with edge_index shape: {edge_index.shape}")

        # Simulate GNN usage where edge_index must be (2, E)
        if edge_index.shape[0] != 2:
            print(
                f"ISSUE DETECTED: edge_index shape is {edge_index.shape}, expected (2, E)"
            )
            # In real GNN, this might crash or produce wrong output
            # For reproduction, we can just error out or let it slide to see if torch.cat errors later

        # Simulate some processing
        out = self.linear(x)
        return out


def run_reproduction():
    model = MockGNNModel()
    dl = IntegratedGradients(model)

    # Scenario: N=2 nodes, F=10 features.
    N = 2
    F = 10
    x = torch.randn(N, F, requires_grad=True)

    # edge_index: (2, E). Let's say 1 edge.
    edge_index = torch.tensor([[0], [1]])
    print(f"Original x shape: {x.shape}")
    print(f"Original edge_index shape: {edge_index.shape}")

    try:
        # IntegratedGradients default n_steps=50.
        # It expands x to (50*2, F) = (100, F).
        # If it expands edge_index, it becomes (100, E).
        attribution = dl.attribute(
            x, target=0, n_steps=50, additional_forward_args=(edge_index,)
        )
        print("Execution finished without error.")
    except Exception as e:
        print(f"\nCaught exception: {e}")


if __name__ == "__main__":
    run_reproduction()
