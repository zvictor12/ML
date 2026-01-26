using ButterflySite.Models;
using Microsoft.Extensions.Hosting;
using System.Text.Json;
using System.Text.RegularExpressions;
using System.Numerics;

namespace ButterflySite.Services;

// Простой локальный сервис поиска по embeddings.npy + metadata.jsonl + data/<species>/*
// Эвристика для выбора файла изображения: fileIndex = (id / 5) % countFiles
public sealed class LocalSearchService
{
    private readonly float[][] _embeddings;
    private readonly double[] _norms;
    private readonly Dictionary<int, (string species, string? imageUrl)> _metadata;
    private readonly string _dataRoot;

    public LocalSearchService(IHostEnvironment env)
    {
        _dataRoot = Path.Combine(env.ContentRootPath, "data");

        var embeddingsPath = Path.Combine(env.ContentRootPath, "embeddings.npy");
        if (!File.Exists(embeddingsPath))
        {
            _embeddings = Array.Empty<float[]>();
            _norms = Array.Empty<double>();
        }
        else
        {
            _embeddings = LoadNpy(embeddingsPath);
            _norms = new double[_embeddings.Length];
            for (int i = 0; i < _embeddings.Length; i++)
            {
                _norms[i] = Math.Sqrt(_embeddings[i].Select(v => (double)v * v).Sum());
                if (_norms[i] == 0) _norms[i] = 1e-6;
            }
        }

        var metadataPath = Path.Combine(env.ContentRootPath, "metadata.jsonl");
        _metadata = new Dictionary<int, (string species, string? imageUrl)>();
        if (File.Exists(metadataPath))
        {
            foreach (var line in File.ReadLines(metadataPath))
            {
                if (string.IsNullOrWhiteSpace(line)) continue;
                try
                {
                    using var doc = JsonDocument.Parse(line);
                    var root = doc.RootElement;
                    if (!root.TryGetProperty("id", out var idProp)) continue;
                    int id = idProp.GetInt32();
                    string species = root.TryGetProperty("species", out var sp) ? sp.GetString() ?? "Unknown" : "Unknown";
                    string? imageUrl = root.TryGetProperty("image_url", out var iu) && iu.ValueKind != JsonValueKind.Null ? iu.GetString() : null;
                    _metadata[id] = (species, imageUrl);
                }
                catch
                {
                    // ignore malformed lines
                }
            }
        }
    }

    public Task<IReadOnlyList<SimilarityResult>> SearchAsync(float[] vector, int limit)
    {
        var results = new List<(int idx, double score)>();

        if (_embeddings.Length == 0)
            return Task.FromResult<IReadOnlyList<SimilarityResult>>(Array.Empty<SimilarityResult>());

        double qnorm = Math.Sqrt(vector.Select(v => (double)v * v).Sum());
        if (qnorm == 0) qnorm = 1e-6;

        for (int i = 0; i < _embeddings.Length; i++)
        {
            // cosine similarity
            double dot = 0;
            var emb = _embeddings[i];
            int len = Math.Min(emb.Length, vector.Length);
            for (int j = 0; j < len; j++) dot += emb[j] * vector[j];
            double score = dot / (_norms[i] * qnorm);
            results.Add((i, score));
        }

        var top = results.OrderByDescending(r => r.score).Take(limit).ToList();

        var outList = new List<SimilarityResult>(top.Count);
        foreach (var (idx, score) in top)
        {
            string species = _metadata.TryGetValue(idx, out var md) ? md.species : "Unknown";
            string? imageUrl = _metadata.TryGetValue(idx, out md) && md.imageUrl is not null ? md.imageUrl : FindLocalImageUrl(idx, species);
            outList.Add(new SimilarityResult(species, imageUrl, (float)score));
        }

        return Task.FromResult<IReadOnlyList<SimilarityResult>>(outList);
    }

    private string? FindLocalImageUrl(int id, string species)
    {
        try
        {
            var dir = Path.Combine(_dataRoot, species);
            if (!Directory.Exists(dir)) return null;
            var files = Directory.GetFiles(dir)
                                 .Where(f => !f.EndsWith(".json") && !f.EndsWith(".npz"))
                                 .OrderBy(f => f, StringComparer.OrdinalIgnoreCase)
                                 .ToArray();
            if (files.Length == 0) return null;
            int imageIndex = (id / 5) % files.Length; // эвристика: по 5 эмбеддингов на одно изображение
            var fname = Path.GetFileName(files[imageIndex]);
            // Static files будут раздаваться из data/ под /images
            return $"/images/{Uri.EscapeDataString(species)}/{Uri.EscapeDataString(fname)}";
        }
        catch
        {
            return null;
        }
    }

    // Минимальный .npy loader, ожидает dtype float32, shape (N,M) или (N,)
    private static float[][] LoadNpy(string path)
    {
        using var fs = File.OpenRead(path);
        using var br = new BinaryReader(fs);

        // magic
        var magic = br.ReadBytes(6);
        var magicStr = System.Text.Encoding.ASCII.GetString(magic);
        if (!magicStr.StartsWith("\x93NUMPY"))
            throw new InvalidDataException("Not a .npy file");

        byte major = br.ReadByte();
        byte minor = br.ReadByte();

        int headerLen;
        if (major == 1)
        {
            headerLen = br.ReadUInt16(); // little-endian
        }
        else
        {
            headerLen = (int)br.ReadUInt32();
        }

        var headerBytes = br.ReadBytes(headerLen);
        var header = System.Text.Encoding.ASCII.GetString(headerBytes).Trim();

        // Extract descr and shape
        var descrMatch = Regex.Match(header, @"'descr':\s*'(?<d>[^']*)'");
        if (!descrMatch.Success) descrMatch = Regex.Match(header, "\"descr\":\\s*\"(?<d>[^\"]*)\"");
        var descr = descrMatch.Success ? descrMatch.Groups["d"].Value : throw new InvalidDataException("No descr in header");
        var littleEndian = descr.StartsWith("<") || descr.StartsWith("|");
        var dtype = descr.TrimStart('<', '>', '|');

        if (!dtype.StartsWith("f4") && !dtype.StartsWith("f"))
            throw new NotSupportedException($"Unsupported dtype {descr}");

        var shapeMatch = Regex.Match(header, @"'shape':\s*\((?<s>[^\)]*)\)");
        if (!shapeMatch.Success) shapeMatch = Regex.Match(header, "\"shape\":\\s*\\((?<s>[^\\)]*)\\)");
        if (!shapeMatch.Success) throw new InvalidDataException("No shape in header");
        var shapeStr = shapeMatch.Groups["s"].Value;
        var shapeParts = shapeStr.Split(',', StringSplitOptions.RemoveEmptyEntries).Select(s => s.Trim()).ToArray();
        if (shapeParts.Length == 0) throw new InvalidDataException("Cannot parse shape");
        int dim0 = int.Parse(shapeParts[0]);
        int dim1 = shapeParts.Length > 1 ? int.Parse(shapeParts[1]) : 1;

        var total = dim0 * dim1;
        var data = new float[total];
        for (int i = 0; i < total; i++)
        {
            // BinaryReader reads in little-endian by default on little-endian machines.
            data[i] = br.ReadSingle();
        }

        var result = new float[dim0][];
        for (int i = 0; i < dim0; i++)
        {
            var row = new float[dim1];
            Array.Copy(data, i * dim1, row, 0, dim1);
            result[i] = row;
        }

        return result;
    }
}