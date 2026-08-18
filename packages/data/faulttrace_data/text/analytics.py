from typing import Any

from faulttrace_core.retrieval import TextDocument
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer


class CorpusAnalytics:
    """Exploratory text-mining analytics on a corpus."""

    def __init__(self, docs: list[TextDocument]):
        self.docs = docs
        self.texts = [doc.text for doc in self.docs if doc.text]

    def compute_tfidf_terms(self, top_n: int = 20) -> list[dict[str, Any]]:
        """Compute top TF-IDF terms across the entire corpus."""
        if not self.texts:
            return []

        vectorizer = TfidfVectorizer(stop_words='english', max_features=1000)
        tfidf_matrix = vectorizer.fit_transform(self.texts)

        # Sum tfidf scores across all documents
        sum_tfidf = tfidf_matrix.sum(axis=0)

        # Get feature names
        feature_names = vectorizer.get_feature_names_out()

        # Pair feature names with their scores
        term_scores = [(feature_names[col], sum_tfidf[0, col]) for col in range(sum_tfidf.shape[1])]

        # Sort by score
        term_scores.sort(key=lambda x: x[1], reverse=True)

        return [{"term": term, "score": float(score)} for term, score in term_scores[:top_n]]

    def compute_ngrams(self, n: int = 2, top_n: int = 20) -> list[dict[str, Any]]:
        """Compute most frequent n-grams."""
        if not self.texts:
            return []

        vectorizer = CountVectorizer(ngram_range=(n, n), stop_words='english')
        count_matrix = vectorizer.fit_transform(self.texts)

        sum_counts = count_matrix.sum(axis=0)
        feature_names = vectorizer.get_feature_names_out()

        term_counts = [(feature_names[col], sum_counts[0, col]) for col in range(sum_counts.shape[1])]
        term_counts.sort(key=lambda x: x[1], reverse=True)

        return [{"ngram": term, "count": int(count)} for term, count in term_counts[:top_n]]

    def compute_clusters(self, n_clusters: int = 5) -> list[dict[str, Any]]:
        """Cluster documents using KMeans on TF-IDF features."""
        if not self.texts or len(self.texts) < n_clusters:
            return []

        vectorizer = TfidfVectorizer(stop_words='english', max_features=500)
        tfidf_matrix = vectorizer.fit_transform(self.texts)

        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        labels = kmeans.fit_predict(tfidf_matrix)

        # Group documents by cluster
        clusters = {i: [] for i in range(n_clusters)}
        for doc_idx, label in enumerate(labels):
            clusters[label].append(self.docs[doc_idx].doc_id)

        # Get top terms for each cluster center
        order_centroids = kmeans.cluster_centers_.argsort()[:, ::-1]
        terms = vectorizer.get_feature_names_out()

        results = []
        for i in range(n_clusters):
            top_terms = [terms[ind] for ind in order_centroids[i, :5]]
            results.append({
                "cluster_id": i,
                "size": len(clusters[i]),
                "top_terms": top_terms,
                "document_ids": clusters[i][:10] # Return up to 10 sample doc IDs
            })

        return results

    def get_summary_stats(self) -> dict[str, Any]:
        """Basic corpus statistics."""
        if not self.texts:
            return {"doc_count": 0, "avg_length": 0, "total_words": 0}

        word_counts = [len(text.split()) for text in self.texts]

        return {
            "doc_count": len(self.docs),
            "total_words": sum(word_counts),
            "avg_length": sum(word_counts) / len(word_counts) if word_counts else 0,
            "max_length": max(word_counts) if word_counts else 0,
            "min_length": min(word_counts) if word_counts else 0
        }
