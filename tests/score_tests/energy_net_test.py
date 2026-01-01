import pytest
import torch
import torch.nn as nn
from score_nets.graph_energy_net import (
    EnergyNet,
    GraphTransformer,
    NodeInitializer,
    TimeEmbedding,
    compute_edge_features
)


@pytest.fixture
def device():
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')


@pytest.fixture
def single_molecule_data():
    num_atoms = 10
    atom_dim = 5
    data = {
        'atom_features': torch.randn(num_atoms, atom_dim),
        'coords': torch.randn(num_atoms, 3),
        'edge_index': torch.randint(0, num_atoms, (2, 30)),
        'time': torch.tensor(0.5),
        'num_atoms': num_atoms,
        'atom_dim': atom_dim
    }
    return data


@pytest.fixture
def batched_molecule_data():
    num_atoms = 12
    atom_dim = 5
    batch = torch.cat([
        torch.zeros(5, dtype=torch.long),
        torch.ones(7, dtype=torch.long)
    ])
    edge_index_mol1 = torch.randint(0, 5, (2, 10))
    edge_index_mol2 = torch.randint(5, 12, (2, 15))
    edge_index = torch.cat([edge_index_mol1, edge_index_mol2], dim=1)
    time = torch.tensor([0.3, 0.7])
    data = {
        'atom_features': torch.randn(num_atoms, atom_dim),
        'coords': torch.randn(num_atoms, 3),
        'edge_index': edge_index,
        'time': time,
        'batch': batch,
        'num_atoms': num_atoms,
        'atom_dim': atom_dim
    }
    return data


@pytest.fixture
def model_params():
    return {
        'atom_dim': 5,
        'hidden_dim': 64,
        'num_layers': 3,
        'dropout': 0.1
    }

class TestModelInstantiation:
    def test_score_net_creation(self, model_params):
        model = EnergyNet(**model_params)
        assert isinstance(model, nn.Module)
        assert len(model.layers) == model_params['num_layers']

    def test_graph_transformer_creation(self):
        layer = GraphTransformer(hidden_dim=64, dropout=0.1)
        assert isinstance(layer, nn.Module)

    def test_node_initializer_creation(self):
        initializer = NodeInitializer(atom_dim=5, hidden_dim=64)
        assert isinstance(initializer, nn.Module)

    def test_time_embedding_creation(self):
        time_emb = TimeEmbedding(embed_dim=64)
        assert isinstance(time_emb, nn.Module)


class TestForwardPass:
    def test_single_molecule_forward(self, single_molecule_data, model_params):
        model = EnergyNet(**model_params)
        model.eval()
        with torch.no_grad():
            scores = model(
                single_molecule_data['coords'],
                single_molecule_data['atom_features'],
                single_molecule_data['edge_index'],
                single_molecule_data['time']
            )
        assert scores.shape == (single_molecule_data['num_atoms'], 3)
        assert not torch.isnan(scores).any(), "Output contains NaN values"
        assert not torch.isinf(scores).any(), "Output contains inf values"

    def test_batched_molecule_forward(self, batched_molecule_data, model_params):
        model = EnergyNet(**model_params)
        model.eval()
        with torch.no_grad():
            scores = model(
                batched_molecule_data['coords'],
                batched_molecule_data['atom_features'],
                batched_molecule_data['edge_index'],
                batched_molecule_data['time'],
                batched_molecule_data['batch']
            )
        assert scores.shape == (batched_molecule_data['num_atoms'], 3)
        assert not torch.isnan(scores).any()
        assert not torch.isinf(scores).any()

    def test_backward_pass(self, single_molecule_data, model_params):
        model = EnergyNet(**model_params)
        model.train()
        scores = model(
            single_molecule_data['coords'],
            single_molecule_data['atom_features'],
            single_molecule_data['edge_index'],
            single_molecule_data['time']
        )
        loss = scores.sum()
        loss.backward()
        has_grad = False
        for param in model.parameters():
            if param.grad is not None and param.grad.abs().sum() > 0:
                has_grad = True
                break

        assert has_grad, "No gradients computed in backward pass"

class TestComponents:
    def test_time_embedding_shape(self):
        time_emb = TimeEmbedding(embed_dim=64)
        time_scalar = torch.tensor(0.5)
        out = time_emb(time_scalar)
        assert out.shape == (1, 64)
        time_batch = torch.tensor([0.1, 0.5, 0.9])
        out = time_emb(time_batch)
        assert out.shape == (3, 64)

    def test_node_initializer_shape(self):
        initializer = NodeInitializer(atom_dim=5, hidden_dim=64)
        atom_features = torch.randn(10, 5)
        time = torch.tensor(0.5)
        nodes = initializer(atom_features, time, num_nodes=10)
        assert nodes.shape == (10, 64)

    def test_compute_edge_features(self):
        coords = torch.randn(10, 3)
        edge_index = torch.randint(0, 10, (2, 20))
        edges = compute_edge_features(coords, edge_index)
        assert edges.shape == (20, 4)
        distances = edges[:, 3]
        assert (distances >= 0).all(), "Distances should be non-negative"

    def test_graph_transformer_shape(self):
        layer = GraphTransformer(hidden_dim=64, dropout=0.1)
        nodes = torch.randn(10, 64)
        edges = torch.randn(20, 4)
        edge_index = torch.randint(0, 10, (2, 20))
        out = layer(nodes, edges, edge_index)
        assert out.shape == (10, 64)

class TestTranslationInvariance:
    def test_edge_features_translation_invariant(self):
        coords = torch.randn(10, 3)
        edge_index = torch.randint(0, 10, (2, 20))
        edges1 = compute_edge_features(coords, edge_index)
        translation = torch.randn(1, 3)
        coords_translated = coords + translation
        edges2 = compute_edge_features(coords_translated, edge_index)
        assert torch.allclose(edges1, edges2, atol=1e-6), \
            "Edge features should be translation invariant"

    def test_model_processes_relative_positions(self, single_molecule_data, model_params):
        model = EnergyNet(**model_params)
        model.eval()
        with torch.no_grad():
            scores1 = model(
                single_molecule_data['coords'],
                single_molecule_data['atom_features'],
                single_molecule_data['edge_index'],
                single_molecule_data['time']
            )
        assert scores1.shape == (single_molecule_data['num_atoms'], 3)

class TestPermutationEquivariance:
    def test_permutation_equivariance(self, model_params):
        model = EnergyNet(**model_params)
        model.eval()
        num_atoms = 8
        atom_features = torch.randn(num_atoms, model_params['atom_dim'])
        coords = torch.randn(num_atoms, 3)
        edge_index = torch.randint(0, num_atoms, (2, 15))
        time = torch.tensor(0.5)
        with torch.no_grad():
            scores1 = model(coords, atom_features, edge_index, time)
        perm = torch.randperm(num_atoms)
        inv_perm = torch.argsort(perm)
        atom_features_perm = atom_features[perm]
        coords_perm = coords[perm]
        edge_index_perm = inv_perm[edge_index]
        with torch.no_grad():
            scores2 = model(coords_perm, atom_features_perm, edge_index_perm, time)
        scores2_unperm = scores2[inv_perm]
        assert torch.allclose(scores1, scores2_unperm, atol=1e-4), \
            "Model should be permutation equivariant"

class TestEdgeCases:
    def test_single_atom(self, model_params):
        model = EnergyNet(**model_params)
        model.eval()
        atom_features = torch.randn(1, model_params['atom_dim'])
        coords = torch.randn(1, 3)
        edge_index = torch.zeros((2, 0), dtype=torch.long)  # No edges
        time = torch.tensor(0.5)
        with torch.no_grad():
            scores = model(coords, atom_features, edge_index, time)
        assert scores.shape == (1, 3)
        assert not torch.isnan(scores).any()

    def test_different_time_values(self, single_molecule_data, model_params):
        model = EnergyNet(**model_params)
        model.eval()
        times = [0.0, 0.25, 0.5, 0.75, 1.0]
        for t in times:
            time = torch.tensor(t)
            with torch.no_grad():
                scores = model(
                    single_molecule_data['coords'],
                    single_molecule_data['atom_features'],
                    single_molecule_data['edge_index'],
                    time
                )
            assert scores.shape == (single_molecule_data['num_atoms'], 3)
            assert not torch.isnan(scores).any()

    def test_fully_connected_graph(self, model_params):
        model = EnergyNet(**model_params)
        model.eval()
        num_atoms = 5
        atom_features = torch.randn(num_atoms, model_params['atom_dim'])
        coords = torch.randn(num_atoms, 3)
        edge_index = torch.combinations(torch.arange(num_atoms), r=2).T
        edge_index = torch.cat([edge_index, edge_index.flip(0)], dim=1)  # Bidirectional
        time = torch.tensor(0.5)
        with torch.no_grad():
            scores = model(coords, atom_features, edge_index, time)
        assert scores.shape == (num_atoms, 3)
        assert not torch.isnan(scores).any()

class TestDeviceCompatibility:
    def test_cpu_execution(self, single_molecule_data, model_params):
        model = EnergyNet(**model_params).cpu()
        model.eval()
        coords = single_molecule_data['coords'].cpu()
        atom_features = single_molecule_data['atom_features'].cpu()
        edge_index = single_molecule_data['edge_index'].cpu()
        time = single_molecule_data['time'].cpu()
        with torch.no_grad():
            scores = model(coords, atom_features, edge_index, time)
        assert scores.device.type == 'cpu'
        assert scores.shape == (single_molecule_data['num_atoms'], 3)

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_cuda_execution(self, single_molecule_data, model_params):
        model = EnergyNet(**model_params).cuda()
        model.eval()
        coords = single_molecule_data['coords'].cuda()
        atom_features = single_molecule_data['atom_features'].cuda()
        edge_index = single_molecule_data['edge_index'].cuda()
        time = single_molecule_data['time'].cuda()
        with torch.no_grad():
            scores = model(coords, atom_features, edge_index, time)
        assert scores.device.type == 'cuda'
        assert scores.shape == (single_molecule_data['num_atoms'], 3)

class TestReproducibility:
    def test_deterministic_forward_pass(self, single_molecule_data, model_params):
        torch.manual_seed(42)
        model1 = EnergyNet(**model_params)
        model1.eval()
        torch.manual_seed(42)
        model2 = EnergyNet(**model_params)
        model2.eval()
        model2.load_state_dict(model1.state_dict())
        with torch.no_grad():
            scores1 = model1(
                single_molecule_data['coords'],
                single_molecule_data['atom_features'],
                single_molecule_data['edge_index'],
                single_molecule_data['time']
            )
            scores2 = model2(
                single_molecule_data['coords'],
                single_molecule_data['atom_features'],
                single_molecule_data['edge_index'],
                single_molecule_data['time']
            )
        assert torch.allclose(scores1, scores2, atol=1e-6), \
            "Model should produce identical outputs with same weights"



if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])