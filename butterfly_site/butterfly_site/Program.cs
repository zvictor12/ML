using ButterflySite.Models;
using ButterflySite.Services;
using Microsoft.Extensions.FileProviders;

var builder = WebApplication.CreateBuilder(args);

builder.Services.Configure<ModelOptions>(builder.Configuration.GetSection(ModelOptions.SectionName));
builder.Services.Configure<QdrantOptions>(builder.Configuration.GetSection(QdrantOptions.SectionName));
builder.Services.Configure<SearchOptions>(builder.Configuration.GetSection(SearchOptions.SectionName));

builder.Services.AddSingleton<EmbeddingService>();
builder.Services.AddSingleton<QdrantSearchService>();
builder.Services.AddSingleton<LocalSearchService>();

var app = builder.Build();

var logger = app.Logger;

// Раздаём папку data под /images, чтобы локальные картинки были доступны.
var imagesPath = Path.Combine(app.Environment.ContentRootPath, "data");
if (Directory.Exists(imagesPath))
{
    app.UseStaticFiles(new StaticFileOptions
    {
        FileProvider = new PhysicalFileProvider(imagesPath),
        RequestPath = "/images"
    });
}
else
{
    logger.LogWarning("Folder 'data' not found at {Path}. Local images won't be available.", imagesPath);
}

app.MapGet("/", () => Results.Content(HtmlTemplates.UploadForm, "text/html; charset=utf-8"));

app.MapPost("/search", async (
    HttpRequest request,
    EmbeddingService embeddingService,
    QdrantSearchService qdrant,
    LocalSearchService local,
    Microsoft.Extensions.Options.IOptions<SearchOptions> searchOpt) =>
{
    if (!request.HasFormContentType)
        return Results.BadRequest("Expected multipart form data.");

    var form = await request.ReadFormAsync();
    var file = form.Files.FirstOrDefault();
    if (file is null || file.Length == 0)
        return Results.BadRequest("Upload an image file.");

    await using var stream = file.OpenReadStream();
    var embedding = await embeddingService.GetEmbeddingAsync(stream);

    IReadOnlyList<SimilarityResult> results;

    // Режим поиска: local | qdrant | auto
    var mode = (searchOpt.Value.Mode ?? "auto").ToLowerInvariant();

    if (mode == "local")
    {
        results = await local.SearchAsync(embedding, limit: 5);
    }
    else if (mode == "qdrant")
    {
        results = await qdrant.SearchAsync(embedding, limit: 5);
    }
    else // auto
    {
        try
        {
            results = await qdrant.SearchAsync(embedding, limit: 5);
        }
        catch (Exception ex)
        {
            logger.LogWarning(ex, "Qdrant search failed, falling back to local search.");
            results = await local.SearchAsync(embedding, limit: 5);
        }
    }

    // Предсказанный класс — большинство среди топ-5 (как у тебя сейчас)
    string predictedSpecies = "Unknown";
    if (results.Count > 0)
    {
        var grouped = results
            .GroupBy(r => r.Species ?? "Unknown")
            .Select(g => new { Species = g.Key, Count = g.Count(), ScoreSum = g.Sum(x => x.Score) })
            .OrderByDescending(x => x.Count)
            .ThenByDescending(x => x.ScoreSum)
            .First();

        predictedSpecies = grouped.Species;
    }

    var page = HtmlTemplates.RenderResults(results, predictedSpecies);
    return Results.Content(page, "text/html; charset=utf-8");
});

app.MapGet("/health", (Microsoft.Extensions.Options.IOptions<ModelOptions> m) =>
{
    var modelOk = File.Exists(m.Value.OnnxPath);
    return Results.Json(new
    {
        ok = modelOk,
        modelPath = m.Value.OnnxPath,
        modelExists = modelOk,
        dataExists = Directory.Exists(imagesPath)
    });
});

app.Run();

static class HtmlTemplates
{
    public static string UploadForm => """
<!doctype html>
<html lang="ru">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Поиск бабочек</title>
  <style>
    body { font-family: Arial, sans-serif; max-width: 840px; margin: 2rem auto; padding: 0 1rem; }
    .results { display: grid; grid-template-columns: repeat(auto-fill, minmax(160px, 1fr)); gap: 1rem; }
    .card { border: 1px solid #e0e0e0; border-radius: 8px; padding: 0.5rem; }
    .card img { max-width: 100%; height: auto; border-radius: 6px; }
    .label { font-weight: 600; margin-top: 0.5rem; }
    .predicted { font-size: 1.1rem; font-weight: 700; margin-bottom: 1rem; }
  </style>
</head>
<body>
  <h1>Поиск бабочек по фото</h1>
  <p>Загрузите фотографию, чтобы получить вид и 5 ближайших изображений.</p>
  <form method="post" action="/search" enctype="multipart/form-data">
    <input type="file" name="file" accept="image/*" required />
    <button type="submit">Найти</button>
  </form>
</body>
</html>
""";

    public static string RenderResults(IReadOnlyList<SimilarityResult> results, string predictedSpecies)
    {
        var cards = string.Join("\n", results.Select(result =>
        {
            var imageHtml = string.IsNullOrWhiteSpace(result.ImageUrl)
                ? "<div class=\"placeholder\">Нет изображения</div>"
                : $"<img src=\"{result.ImageUrl}\" alt=\"{result.Species}\" />";

            return $"""
<div class="card">
  {imageHtml}
  <div class="label">{result.Species}</div>
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
    .predicted {{ font-size: 1.25rem; font-weight: 800; margin-bottom: 1rem; }}
  </style>
</head>
<body>
  <a href=""/"">← Назад</a>
  <h1>Результаты поиска</h1>
  <div class=""predicted"">Предположительный вид: {System.Net.WebUtility.HtmlEncode(predictedSpecies)}</div>
  <div class=""results"">
    {cards}
  </div>
</body>
</html>";
    }
}

public sealed class SearchOptions
{
    public const string SectionName = "Search";
    public string? Mode { get; init; } = "auto";
}
