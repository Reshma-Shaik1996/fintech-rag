using Microsoft.Extensions.Configuration;
using Microsoft.SemanticKernel;
using Microsoft.SemanticKernel.ChatCompletion;

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