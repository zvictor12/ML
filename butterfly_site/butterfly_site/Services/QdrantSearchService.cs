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
            var species =
                GetPayloadString(point.Payload, "species") ?? "Unknown";

            var imageUrl = GetPayloadString(point.Payload, "image_url");

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
            _ => null
        };
    }
}

