using FintechRag.Console.Ingestion;

namespace FintechRag.Console.Retrieval;

/// <summary>
/// Combines vector and BM25 results via Reciprocal Rank Fusion:
/// score = sum over lists of 1 / (k + rank). Rank-based, so the
/// incompatible score scales of the two methods never matter.
/// </summary>
public class HybridSearcher(VectorStore vectorStore, Bm25Index bm25)
{
    private const int RrfK = 60; // standard damping constant from the RRF paper

    public List<SearchResult> Search(string query, float[] queryVector, int topK = 5)
    {
        // Pull a deeper candidate pool from each method than we plan to return
        var vectorResults = vectorStore.Search(queryVector, topK * 3);
        var bm25Results = bm25.Search(query, topK * 3);

        var fused = new Dictionary<string, (DocumentChunk Chunk, double Score)>();

        void Accumulate(List<SearchResult> results)
        {
            for (int rank = 0; rank < results.Count; rank++)
            {
                var chunk = results[rank].Chunk;
                double rrf = 1.0 / (RrfK + rank + 1);
                fused[chunk.Id] = fused.TryGetValue(chunk.Id, out var existing)
                    ? (chunk, existing.Score + rrf)
                    : (chunk, rrf);
            }
        }

        Accumulate(vectorResults);
        Accumulate(bm25Results);

        return fused.Values
            .OrderByDescending(x => x.Score)
            .Take(topK)
            .Select(x => new SearchResult(x.Chunk, x.Score))
            .ToList();
    }
}