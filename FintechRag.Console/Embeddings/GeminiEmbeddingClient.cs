using System.Net.Http.Json;
using System.Text.Json.Serialization;

namespace FintechRag.Console.Embeddings;

public class GeminiEmbeddingClient(string apiKey)
{
      private const string Model = "gemini-embedding-001";
    private const string Url =
        $"https://generativelanguage.googleapis.com/v1beta/models/{Model}:batchEmbedContents";
    private const int BatchSize = 25; // API max per request // smaller gulps for the free tier
    private const int MaxRetries = 5;


    private readonly HttpClient _http = new();

    /// <summary>Embeds texts in batches. Returns one vector (768 floats) per input text.</summary>
public async Task<List<float[]>> EmbedAsync(IReadOnlyList<string> texts)
{
    var allVectors = new List<float[]>(texts.Count);

    for (int i = 0; i < texts.Count; i += BatchSize)
    {
        var batch = texts.Skip(i).Take(BatchSize).ToList();
        var delay = TimeSpan.FromSeconds(5);

        for (int attempt = 1; ; attempt++)
        {
            // Request must be rebuilt each attempt — HttpRequestMessage is single-use
            var request = new HttpRequestMessage(HttpMethod.Post, Url);
            request.Headers.Add("x-goog-api-key", apiKey);
            request.Content = JsonContent.Create(new
            {
                requests = batch.Select(t => new
                {
                    model = $"models/{Model}",
                    content = new { parts = new[] { new { text = t } } },
                    outputDimensionality = 768
                })
            });

            var response = await _http.SendAsync(request);

            if (response.StatusCode == System.Net.HttpStatusCode.TooManyRequests
                && attempt < MaxRetries)
            {
                System.Console.WriteLine(
                    $"Rate limited (429). Waiting {delay.TotalSeconds}s before retry {attempt}/{MaxRetries}...");
                await Task.Delay(delay);
                delay *= 2; // exponential backoff: 5s, 10s, 20s, 40s
                continue;
            }

            response.EnsureSuccessStatusCode();

            var result = await response.Content.ReadFromJsonAsync<BatchEmbedResponse>()
                ?? throw new InvalidOperationException("Empty embedding response.");

            allVectors.AddRange(result.Embeddings.Select(e => e.Values));
            break;
        }

        System.Console.WriteLine($"Embedded {Math.Min(i + BatchSize, texts.Count)}/{texts.Count} chunks...");
        await Task.Delay(TimeSpan.FromSeconds(2)); // gentle pacing between batches
    }

    return allVectors;
}
    private record BatchEmbedResponse(
        [property: JsonPropertyName("embeddings")] List<EmbeddingValues> Embeddings);
    private record EmbeddingValues(
        [property: JsonPropertyName("values")] float[] Values);
}