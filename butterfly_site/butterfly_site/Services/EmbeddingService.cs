using ButterflySite.Models;
using Microsoft.Extensions.Options;
using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using SixLabors.ImageSharp;
using SixLabors.ImageSharp.PixelFormats;
using SixLabors.ImageSharp.Processing;

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
        await using var copy = new MemoryStream();
        await imageStream.CopyToAsync(copy);
        copy.Position = 0;

        using var loaded = await Image.LoadAsync<Rgb24>(copy);

        const int target = 224;

        Image<Rgb24> img = loaded.Clone();

        if (img.Width > target || img.Height > target)
        {
            img.Mutate(x => x.Resize(new ResizeOptions
            {
                Size = new Size(target, target),
                Mode = ResizeMode.Max
            }));
        }

        if (img.Width != target || img.Height != target)
        {
            var padded = PadToSizeCenter(img, target, target);
            img.Dispose();
            img = padded;
        }

        // NHWC: [1, H, W, 3]
        var inputTensor = CreateInputTensorNHWC(img);

        var session = _session.Value;
        var inputName = session.InputMetadata.Keys.First();

        var named = NamedOnnxValue.CreateFromTensor(inputName, inputTensor);
        using var results = session.Run(new[] { named });

        var first = results.First();
        var outTensor = first.AsTensor<float>();
        var outArray = outTensor.ToArray();

        img.Dispose();

        if (outArray.Length == _options.EmbeddingSize)
        {
            L2NormalizeInPlace(outArray);
            return outArray;
        }

        var embedding = new float[_options.EmbeddingSize];
        Array.Copy(outArray, embedding, Math.Min(outArray.Length, embedding.Length));

        L2NormalizeInPlace(embedding);
        return embedding;
    }

    private static DenseTensor<float> CreateInputTensorNHWC(Image<Rgb24> image)
    {
        int height = image.Height;
        int width = image.Width;

        var tensor = new DenseTensor<float>(new[] { 1, height, width, 3 });

        for (int y = 0; y < height; y++)
        {
            for (int x = 0; x < width; x++)
            {
                var p = image[x, y];

                float r = p.R / 255f;
                float g = p.G / 255f;
                float b = p.B / 255f;

                tensor[0, y, x, 0] = r;
                tensor[0, y, x, 1] = g;
                tensor[0, y, x, 2] = b;
            }
        }

        return tensor;
    }

    private static void L2NormalizeInPlace(float[] v)
    {
        double sum = 0;
        for (int i = 0; i < v.Length; i++)
            sum += (double)v[i] * v[i];

        var norm = Math.Sqrt(sum);
        if (norm < 1e-12)
            return;

        var inv = 1.0 / norm;
        for (int i = 0; i < v.Length; i++)
            v[i] = (float)(v[i] * inv);
    }

    private static Image<Rgb24> PadToSizeCenter(Image<Rgb24> src, int dstW, int dstH)
    {
        var dst = new Image<Rgb24>(dstW, dstH, new Rgb24(0, 0, 0));

        int offsetX = (dstW - src.Width) / 2;
        int offsetY = (dstH - src.Height) / 2;

        dst.Mutate(ctx => ctx.DrawImage(src, new Point(offsetX, offsetY), 1f));
        return dst;
    }
}
