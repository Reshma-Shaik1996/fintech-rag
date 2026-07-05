using Microsoft.Extensions.Configuration;
using Microsoft.SemanticKernel;
using Microsoft.SemanticKernel.ChatCompletion;
using FintechRag.Console.Ingestion;
using FintechRag.Console.Embeddings;
using System.Text.Json;
using FintechRag.Console.Retrieval;


// 1. Load the API key from user secrets (never from code or the repo)
var config = new ConfigurationBuilder()
    .AddUserSecrets<Program>()
    .Build();

var apiKey = config["Gemini:ApiKey"]
    ?? throw new InvalidOperationException(
        "API key missing. Run: dotnet user-secrets set \"Gemini:ApiKey\" \"your-key\"");


// Phase 2 test: load the annual report
var pdfPath = Path.Combine("data", "annual-report.pdf");
var pages = DocumentLoader.LoadPdf(pdfPath);

Console.WriteLine($"Loaded {pages.Count} pages.");
Console.WriteLine($"--- Sample from page 1 ---");
Console.WriteLine(pages[0].Text[..Math.Min(500, pages[0].Text.Length)]);
Console.WriteLine("---------------------------\n");

//Phase 3A:TextChunker test: split the pages into chunks for RAG
var chunks = TextChunker.ChunkPages(pages);
Console.WriteLine($"Created {chunks.Count} chunks from {pages.Count} pages.");
Console.WriteLine($"--- Sample chunk ({chunks[10].Id}) ---");
Console.WriteLine(chunks[10].Text);
Console.WriteLine("---------------------------\n");

// Phase 3B: embed chunks (cached — only calls the API if no cache file exists)
var embeddingsPath = Path.Combine("data", "embeddings.json");
List<float[]> vectors;

if (File.Exists(embeddingsPath))
{
    vectors = JsonSerializer.Deserialize<List<float[]>>(File.ReadAllText(embeddingsPath))!;
    Console.WriteLine($"Loaded {vectors.Count} cached embeddings.");
}
else
{
    var embedder = new GeminiEmbeddingClient(apiKey);
    vectors = await embedder.EmbedAsync(chunks.Select(c => c.Text).ToList());
    File.WriteAllText(embeddingsPath, JsonSerializer.Serialize(vectors));
    Console.WriteLine($"Embedded and cached {vectors.Count} vectors.");
}

Console.WriteLine($"Each vector has {vectors[0].Length} dimensions.");
Console.WriteLine($"First 5 numbers of chunk 0's vector: [{string.Join(", ", vectors[0].Take(5))}]\n");


// Phase 4: create a vector store for retrieval // test: semantic search
var store = new VectorStore(chunks, vectors);
var embedderForQueries = new GeminiEmbeddingClient(apiKey);

var testQuery = "How much did the bank lose to bad loans?";
Console.WriteLine($"Query: \"{testQuery}\"\n");

var queryVector = (await embedderForQueries.EmbedAsync([testQuery]))[0];
var results = store.Search(queryVector, topK: 3);

foreach (var r in results)
{
    Console.WriteLine($"[{r.Chunk.Id}] score: {r.Score:F4} (page {r.Chunk.PageNumber})");
    Console.WriteLine(r.Chunk.Text[..Math.Min(200, r.Chunk.Text.Length)] + "...\n");
}


//
// 2. Build the kernel — the central object Semantic Kernel routes everything through
#pragma warning disable SKEXP0070 // Gemini connector is marked experimental
var kernel = Kernel.CreateBuilder()
    .AddGoogleAIGeminiChatCompletion(
        modelId: "gemini-2.5-flash",
        apiKey: apiKey)
    .Build();
#pragma warning restore SKEXP0070

// 3. Get the chat service and keep a running conversation history
var chat = kernel.GetRequiredService<IChatCompletionService>();
var history = new ChatHistory();
history.AddSystemMessage("You are an assistant specializing in financial document analysis.");

Console.WriteLine("Fintech RAG — Phase 1: Kernel is alive. Ask something (type 'exit' to quit).");

while (true)
{
    Console.Write("\nYou: ");
    var input = Console.ReadLine();

    if (string.IsNullOrWhiteSpace(input)) continue;
    if (input.Equals("exit", StringComparison.OrdinalIgnoreCase)) break;

    history.AddUserMessage(input);

    try
    {
        var response = await chat.GetChatMessageContentAsync(history, kernel: kernel);
        Console.WriteLine($"\nAI: {response.Content}");
        history.AddAssistantMessage(response.Content ?? string.Empty);
    }
    catch (Exception ex) when (ex.Message.Contains("503") || ex.Message.Contains("429"))
    {
        Console.WriteLine("\n[Gemini is briefly overloaded — wait a few seconds and ask again.]");
}
}