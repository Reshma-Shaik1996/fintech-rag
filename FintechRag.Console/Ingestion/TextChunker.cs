namespace FintechRag.Console.Ingestion;

/// <summary>A chunk of document text, with provenance for citations later.</summary>
public record DocumentChunk(string Id, int PageNumber, int ChunkIndex, string Text);

public static class TextChunker
{
    /// <summary>
    /// Splits pages into overlapping chunks, preferring sentence boundaries.
    /// maxChars ~1500 ≈ a few paragraphs. Overlap prevents ideas being
    /// cut in half at chunk borders.
    /// </summary>
    public static List<DocumentChunk> ChunkPages(
        IEnumerable<DocumentPage> pages, int maxChars = 1500, int overlap = 200)
    {
        var chunks = new List<DocumentChunk>();

        foreach (var page in pages)
        {
            var text = page.Text;
            int start = 0, index = 0;

            while (start < text.Length)
            {
                int end = Math.Min(start + maxChars, text.Length);

                // Prefer to break at a sentence end, not mid-sentence
                if (end < text.Length)
                {
                    int sentenceBreak = text.LastIndexOf(". ", end - 1, end - start - 1,
                        StringComparison.Ordinal);
                    if (sentenceBreak > start + maxChars / 2)
                        end = sentenceBreak + 1;
                }

                var slice = text[start..end].Trim();
                if (slice.Length > 0)
                {
                    chunks.Add(new DocumentChunk(
                        Id: $"p{page.PageNumber}-c{index}",
                        PageNumber: page.PageNumber,
                        ChunkIndex: index,
                        Text: slice));
                    index++;
                }

                if (end >= text.Length) break;
                start = Math.Max(end - overlap, start + 1); // overlap, never go backward
            }
        }

        return chunks;
    }
}