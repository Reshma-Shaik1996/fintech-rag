using Microsoft.Extensions.Configuration;
using Microsoft.SemanticKernel;
using Microsoft.SemanticKernel.ChatCompletion;
using FintechRag.Console.Ingestion;

// Phase 2 test: load the annual report
var pdfPath = Path.Combine("data", "annual-report.pdf");
var pages = DocumentLoader.LoadPdf(pdfPath);

Console.WriteLine($"Loaded {pages.Count} pages.");
Console.WriteLine($"--- Sample from page 1 ---");
Console.WriteLine(pages[0].Text[..Math.Min(500, pages[0].Text.Length)]);
Console.WriteLine("---------------------------\n");

//TextChunker test: split the pages into chunks for RAG
var chunks = TextChunker.ChunkPages(pages);
Console.WriteLine($"Created {chunks.Count} chunks from {pages.Count} pages.");
Console.WriteLine($"--- Sample chunk ({chunks[10].Id}) ---");
Console.WriteLine(chunks[10].Text);
Console.WriteLine("---------------------------\n");



// 1. Load the API key from user secrets (never from code or the repo)
var config = new ConfigurationBuilder()
    .AddUserSecrets<Program>()
    .Build();

var apiKey = config["Gemini:ApiKey"]
    ?? throw new InvalidOperationException(
        "API key missing. Run: dotnet user-secrets set \"Gemini:ApiKey\" \"your-key\"");

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

    var response = await chat.GetChatMessageContentAsync(history, kernel: kernel);
    Console.WriteLine($"\nAI: {response.Content}");

    history.AddAssistantMessage(response.Content ?? string.Empty);
}