from infra.load_data.load_aspects import load_aspects

def test_load_aspects():
    aspects = load_aspects()
    assert isinstance(aspects, list)
    assert len(aspects) > 0
    for aspect in aspects:
        assert isinstance(aspect, str)
