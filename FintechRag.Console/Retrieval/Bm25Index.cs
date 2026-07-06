using FintechRag.Console.Ingestion;

namespace FintechRag.Console.Retrieval;

/// <summary>
/// BM25 keyword relevance scoring. Two ingredients:
/// TF (term frequency, with diminishing returns) and
/// IDF (rare terms carry more signal than common ones).
/// </summary>
public class Bm25Index
{
    private const double K1 = 1.5;  // how quickly repeated terms hit diminishing returns
    private const double B = 0.75;  // how much to penalize long chunks

    private readonly IReadOnlyList<DocumentChunk> _chunks;
    private readonly List<Dictionary<string, int>> _termCounts = []; // per-chunk term counts
    private readonly Dictionary<string, int> _docFrequency = [];     // in how many chunks does each term appear
    private readonly double _avgDocLength;

    public Bm25Index(IReadOnlyList<DocumentChunk> chunks)
    {
        _chunks = chunks;

        foreach (var chunk in chunks)
        {
            var counts = Tokenize(chunk.Text)
                .GroupBy(t => t)
                .ToDictionary(g => g.Key, g => g.Count());
            _termCounts.Add(counts);

            foreach (var term in counts.Keys)
                _docFrequency[term] = _docFrequency.GetValueOrDefault(term) + 1;
        }

        _avgDocLength = _termCounts.Average(c => c.Values.Sum());
    }

    public List<SearchResult> Search(string query, int topK = 5)
    {
        var queryTerms = Tokenize(query).Distinct().ToList();
        int n = _chunks.Count;

        var results = new List<SearchResult>();

        for (int i = 0; i < n; i++)
        {
            double score = 0;
            double docLength = _termCounts[i].Values.Sum();

            foreach (var term in queryTerms)
            {
                if (!_termCounts[i].TryGetValue(term, out int tf)) continue;

                int df = _docFrequency[term];
                double idf = Math.Log(1 + (n - df + 0.5) / (df + 0.5));
                double tfNorm = tf * (K1 + 1)
                    / (tf + K1 * (1 - B + B * docLength / _avgDocLength));

                score += idf * tfNorm;
            }

            if (score > 0)
                results.Add(new SearchResult(_chunks[i], score));
        }

        return results.OrderByDescending(r => r.Score).Take(topK).ToList();
    }

    private static List<string> Tokenize(string text) =>
        text.ToLowerInvariant()
            .Split([' ', '\n', '\r', '\t', '.', ',', ';', ':', '(', ')', '$', '%', '"', '\''],
                StringSplitOptions.RemoveEmptyEntries)
            .Where(t => t.Length > 1)
            .ToList();
}
