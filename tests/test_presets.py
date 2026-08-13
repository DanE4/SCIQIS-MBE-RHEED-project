from mbe_rheed_sim import interactive_config, publication_config


def test_named_runtime_presets() -> None:
    interactive = interactive_config(seed=3)
    publication = publication_config(seed=4)
    assert (interactive.lattice_size, interactive.seed) == (16, 3)
    assert (publication.lattice_size, publication.seed) == (64, 4)
