namespace ButterflySite.Models;

public sealed class QdrantOptions
{
    public const string SectionName = "Qdrant";

    public string Url { get; init; } = "http://localhost:6333";
    public string CollectionName { get; init; } = "butterflies";
}
