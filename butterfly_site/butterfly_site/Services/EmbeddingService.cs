using ButterflySite.Models;
using Microsoft.Extensions.Options;
using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using SixLabors.ImageSharp;
using SixLabors.ImageSharp.PixelFormats;

namespace ButterflySite.Services;

public sealed class EmbeddingService
{
    private readonly ModelOptions _options;
    private readonly Lazy<InferenceSession> _session;

    public EmbeddingService(IOptions<ModelOptions> options)
    {
        _options = options.Value;
        _session = new Lazy<InferenceSession>(() => new InferenceSession(_options.OnnxPath));
    }

    public async Task<float[]> GetEmbeddingAsync(Stream imageStream)
    {
        // TODO: Replace placeholder with model-specific preprocessing and ONNX inference.
        // This keeps the site runnable while you wire up the real model inputs/outputs.
        await using var copy = new MemoryStream();
        await imageStream.CopyToAsync(copy);
        copy.Position = 0;

        _ = await Image.LoadAsync<Rgb24>(copy);

        return new float[_options.EmbeddingSize];
    }

    private DenseTensor<float> CreateInputTensor(Image<Rgb24> image)
    {
        // Template for when you wire up ONNX inference.
        return new DenseTensor<float>(new[] { 1, 3, image.Height, image.Width });
    }
}
