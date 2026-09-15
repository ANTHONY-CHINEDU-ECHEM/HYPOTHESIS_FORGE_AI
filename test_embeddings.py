import numpy as np

from hypothesisforge.tools.embeddings import TfidfEmbeddingBackend, cosine_similarity_matrix


def test_tfidf_backend_fit_and_embed_shapes():
    backend = TfidfEmbeddingBackend(max_features=50)
    corpus = [
        "lithium garnet electrolyte ionic conductivity",
        "sulfide argyrodite grain boundary resistance",
        "polymer composite electrolyte mechanical properties",
    ]
    backend.fit(corpus)
    vecs = backend.embed(corpus)
    assert vecs.shape[0] == 3
    assert vecs.shape[1] > 0


def test_tfidf_backend_raises_before_fit():
    backend = TfidfEmbeddingBackend()
    try:
        backend.embed(["hello"])
        assert False, "expected RuntimeError before fit()"
    except RuntimeError:
        pass


def test_cosine_similarity_matrix_self_similarity_is_one():
    vecs = np.array([[1.0, 0.0], [0.0, 1.0]])
    sims = cosine_similarity_matrix(vecs, vecs)
    assert np.allclose(np.diag(sims), 1.0, atol=1e-6)


def test_cosine_similarity_matrix_handles_zero_vectors():
    vecs = np.array([[0.0, 0.0], [1.0, 1.0]])
    sims = cosine_similarity_matrix(vecs, vecs)
    assert not np.isnan(sims).any()


def test_corpus_index_search_returns_ranked_results(corpus):
    results = corpus.search("garnet doped ionic conductivity improvement", top_k=5)
    assert len(results) == 5
    sims = [r.similarity for r in results]
    assert sims == sorted(sims, reverse=True)
    assert all(0.0 <= s <= 1.0 for s in sims)


def test_corpus_index_similarity_stats(corpus):
    stats = corpus.similarity_stats("sulfide argyrodite interfacial stability", top_k=5)
    assert "max_sim" in stats and "mean_sim" in stats
    assert stats["max_sim"] >= stats["mean_sim"]
    assert len(stats["records"]) == 5
