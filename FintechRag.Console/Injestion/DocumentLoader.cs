using UglyToad.PdfPig;

namespace FintechRag.Console.Ingestion;

/// <summary>Extracted text from one page of a document.</summary>
public record DocumentPage(int PageNumber, string Text);

public static class DocumentLoader
{
    /// <summary>Loads a PDF and returns its text, page by page.</summary>
    public static List<DocumentPage> LoadPdf(string filePath)
    {
        if (!File.Exists(filePath))
            throw new FileNotFoundException($"PDF not found at: {filePath}");

        var pages = new List<DocumentPage>();

        using var document = PdfDocument.Open(filePath);
        foreach (var page in document.GetPages())
        {
            var text = page.Text;
            if (!string.IsNullOrWhiteSpace(text))
                pages.Add(new DocumentPage(page.Number, text));
        }

        return pages;
    }
}