using ButterflySite.Models;
using ButterflySite.Services;
using Microsoft.Extensions.Options;

var builder = WebApplication.CreateBuilder(args);

builder.Services.Configure<ModelOptions>(builder.Configuration.GetSection(ModelOptions.SectionName));
builder.Services.Configure<QdrantOptions>(builder.Configuration.GetSection(QdrantOptions.SectionName));

builder.Services.AddSingleton<EmbeddingService>();
builder.Services.AddSingleton<QdrantSearchService>();

var app = builder.Build();

app.UseStaticFiles();

app.MapGet("/", () => Results.Content(HtmlTemplates.UploadForm, "text/html; charset=utf-8"));

app.MapPost("/search", async (HttpRequest request, EmbeddingService embeddingService, QdrantSearchService searchService) =>
{
    if (!request.HasFormContentType)
    {
        return Results.BadRequest("Expected multipart form data.");
    }

    var form = await request.ReadFormAsync();
    var file = form.Files.FirstOrDefault();
    if (file is null || file.Length == 0)
    {
        return Results.BadRequest("Upload an image file.");
    }

    await using var stream = file.OpenReadStream();
    var embedding = await embeddingService.GetEmbeddingAsync(stream);
    var results = await searchService.SearchAsync(embedding, limit: 5);

    var page = HtmlTemplates.RenderResults(results);
    return Results.Content(page, "text/html; charset=utf-8");
});

app.Run();

static class HtmlTemplates
{
    public static string UploadForm => """
<!doctype html>
<html lang=\"ru\">
<head>
  <meta charset=\"utf-8\" />
  <meta name=\"viewport\" content=\"width=device-width, initial-scale=1\" />
  <title>Поиск бабочек</title>
  <style>
    body { font-family: Arial, sans-serif; max-width: 840px; margin: 2rem auto; padding: 0 1rem; }
    .results { display: grid; grid-template-columns: repeat(auto-fill, minmax(160px, 1fr)); gap: 1rem; }
    .card { border: 1px solid #e0e0e0; border-radius: 8px; padding: 0.5rem; }
    .card img { max-width: 100%; height: auto; border-radius: 6px; }
    .label { font-weight: 600; margin-top: 0.5rem; }
  </style>
</head>
<body>
  <h1>Поиск бабочек по фото</h1>
  <p>Загрузите фотографию, чтобы получить вид и 5 ближайших изображений.</p>
  <form method=\"post\" action=\"/search\" enctype=\"multipart/form-data\">
    <input type=\"file\" name=\"file\" accept=\"image/*\" required />
    <button type=\"submit\">Найти</button>
  </form>
</body>
</html>
""";

    public static string RenderResults(IReadOnlyList<SimilarityResult> results)
    {
        var cards = string.Join("\n", results.Select(result =>
        {
            var imageHtml = string.IsNullOrWhiteSpace(result.ImageUrl)
                ? "<div class=\"placeholder\">Нет изображения</div>"
                : $"<img src=\"{result.ImageUrl}\" alt=\"{result.Species}\" />";

            return $"""
<div class=\"card\">
  {imageHtml}
  <div class=\"label\">{result.Species}</div>
  <div>Сходство: {result.Score:F3}</div>
</div>
""";
        }));

        return $@"<!doctype html>
<html lang=""ru"">
<head>
  <meta charset=""utf-8"" />
  <meta name=""viewport"" content=""width=device-width, initial-scale=1"" />
  <title>Результаты</title>
  <style>
    body {{ font-family: Arial, sans-serif; max-width: 900px; margin: 2rem auto; padding: 0 1rem; }}
    .results {{ display: grid; grid-template-columns: repeat(auto-fill, minmax(160px, 1fr)); gap: 1rem; }}
    .card {{ border: 1px solid #e0e0e0; border-radius: 8px; padding: 0.5rem; }}
    .card img {{ max-width: 100%; height: auto; border-radius: 6px; }}
    .label {{ font-weight: 600; margin-top: 0.5rem; }}
  </style>
</head>
<body>
  <a href=""/"">← Назад</a>
  <h1>Результаты поиска</h1>
  <div class=""results"">
    {cards}
  </div>
</body>
</html>";
    }
}