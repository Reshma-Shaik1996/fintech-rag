using FintechRag.Console.Ingestion;

namespace FintechRag.Console.Retrieval;

/// <summary>A chunk paired with its similarity score for a given query.</summary>
public record SearchResult(DocumentChunk Chunk, double Score);

public class VectorStore(IReadOnlyList<DocumentChunk> chunks, IReadOnlyList<float[]> vectors)
{
    /// <summary>Returns the top-k chunks most similar to the query vector.</summary>
    public List<SearchResult> Search(float[] queryVector, int topK = 5)
    {
        return chunks
            .Select((chunk, i) => new SearchResult(chunk, CosineSimilarity(queryVector, vectors[i])))
            .OrderByDescending(r => r.Score)
            .Take(topK)
            .ToList();
    }

    private static double CosineSimilarity(float[] a, float[] b)
    {
        double dot = 0, magA = 0, magB = 0;
        for (int i = 0; i < a.Length; i++)
        {
            dot += a[i] * b[i];
            magA += a[i] * a[i];
            magB += b[i] * b[i];
        }
        return dot / (Math.Sqrt(magA) * Math.Sqrt(magB));
    }
}