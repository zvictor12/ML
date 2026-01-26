using ButterflySite.Models;
using Microsoft.Extensions.Options;
using Qdrant.Client;
using Qdrant.Client.Grpc;

namespace ButterflySite.Services;

public sealed class QdrantSearchService
{
    private readonly QdrantOptions _options;
    private readonly QdrantClient _client;

    public QdrantSearchService(IOptions<QdrantOptions> options)
    {
        _options = options.Value;
        _client = new QdrantClient(new Uri(_options.Url));
    }

    public async Task<IReadOnlyList<SimilarityResult>> SearchAsync(float[] vector, int limit)
    {
        // Exact=true полезно для диагностики (self-search должен давать ~1),
        // и убирает “рандом” approximate HNSW.
        var searchParams = new SearchParams
        {
            Exact = true
        };

        var results = await _client.SearchAsync(
            collectionName: _options.CollectionName,
            vector: vector,
            searchParams: searchParams,
            limit: (ulong)limit,
            payloadSelector: new WithPayloadSelector(true));

        return results.Select(point =>
        {
            // Поддержка двух схем payload:
            // 1) species/image_url (как ожидал код)
            // 2) label (как у тебя в metadata.jsonl)
            var species =
                GetPayloadString(point.Payload, "species") ??
                GetPayloadString(point.Payload, "label") ??
                "Unknown";

            var imageUrl =
                GetPayloadString(point.Payload, "image_url") ??
                GetPayloadString(point.Payload, "img") ??
                GetPayloadString(point.Payload, "path"); // на всякий случай

            return new SimilarityResult(
                Species: species,
                ImageUrl: imageUrl,
                Score: point.Score);
        }).ToList();
    }

    private static string? GetPayloadString(IDictionary<string, Value> payload, string key)
    {
        if (!payload.TryGetValue(key, out var value))
            return null;

        return value.KindCase switch
        {
            Value.KindOneofCase.StringValue => value.StringValue,
            Value.KindOneofCase.IntegerValue => value.IntegerValue.ToString(),
            Value.KindOneofCase.DoubleValue => value.DoubleValue.ToString("G"),
            Value.KindOneofCase.BoolValue => value.BoolValue.ToString(),
            _ => null
        };
    }
}

