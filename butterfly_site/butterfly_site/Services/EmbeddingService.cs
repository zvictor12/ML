using ButterflySite.Models;
using Microsoft.Extensions.Options;
using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using SixLabors.ImageSharp;
using SixLabors.ImageSharp.PixelFormats;
using SixLabors.ImageSharp.Processing;
using System.Numerics;

namespace ButterflySite.Services;

public sealed class EmbeddingService
{
    private readonly ModelOptions _options;
    private readonly Lazy<InferenceSession> _session;

    // ImageNet-like normalization defaults (can be adjusted)
    private static readonly float[] Mean = new float[] { 0.485f, 0.456f, 0.406f };
    private static readonly float[] Std = new float[] { 0.229f, 0.224f, 0.225f };

    public EmbeddingService(IOptions<ModelOptions> options)
    {
        _options = options.Value;
        _session = new Lazy<InferenceSession>(() => new InferenceSession(_options.OnnxPath));
    }

    public async Task<float[]> GetEmbeddingAsync(Stream imageStream)
    {
        // Load image into memory first
        await using var copy = new MemoryStream();
        await imageStream.CopyToAsync(copy);
        copy.Position = 0;

        using var image = await Image.LoadAsync<Rgb24>(copy);

        // Resize to 224x224 (hardcoded per your request)
        const int targetSize = 224;
        image.Mutate(x => x.Resize(new ResizeOptions
        {
            Size = new Size(targetSize, targetSize),
            Mode = ResizeMode.Crop
        }));

        // Ensure session is created and inspect input metadata to know expected layout
        var session = _session.Value;
        var inputName = session.InputMetadata.Keys.First();
        var inputMeta = session.InputMetadata[inputName];
        var dims = inputMeta.Dimensions; // usually length == 4

        // Determine whether model expects NCHW ([1, C, H, W]) or NHWC ([1, H, W, C])
        bool expectsNCHW = false;
        bool expectsNHWC = false;

        if (dims.Length == 4)
        {
            // dims can contain -1 for dynamic. We check which axis equals 3 (channels)
            // Common cases:
            // NCHW: dims = [N, C, H, W] -> dims[1] == 3
            // NHWC: dims = [N, H, W, C] -> dims[3] == 3
            if (dims[1] == 3 || dims[1] == -1 && dims[3] != 3)
                expectsNCHW = true;
            if (dims[3] == 3 || dims[3] == -1 && dims[1] != 3)
                expectsNHWC = true;
        }

        // Fallback heuristics: prefer NHWC if ambiguous (many ONNX vision models use NHWC)
        if (!expectsNCHW && !expectsNHWC)
        {
            // try to infer: if any dim == 224 at index 1 -> probably NHWC; else NCHW
            if (dims.Length == 4 && dims[1] == targetSize) expectsNHWC = true;
            else expectsNCHW = true;
        }

        Tensor<float> inputTensor;
        if (expectsNCHW)
        {
            inputTensor = CreateInputTensorNCHW(image);
        }
        else // NHWC
        {
            inputTensor = CreateInputTensorNHWC(image);
        }

        var named = NamedOnnxValue.CreateFromTensor(inputName, inputTensor);
        using var results = session.Run(new[] { named });

        var first = results.First();
        var outTensor = first.AsTensor<float>();
        var outArray = outTensor.ToArray();

        // Ensure correct size: truncate or pad with zeros
        if (outArray.Length == _options.EmbeddingSize)
            return outArray;

        var embedding = new float[_options.EmbeddingSize];
        Array.Clear(embedding);
        Array.Copy(outArray, embedding, Math.Min(outArray.Length, embedding.Length));
        return embedding;
    }

    private static DenseTensor<float> CreateInputTensorNCHW(Image<Rgb24> image)
    {
        int height = image.Height;
        int width = image.Width;

        var tensor = new DenseTensor<float>(new[] { 1, 3, height, width });

        // NCHW layout: [1, C, H, W]
        for (int y = 0; y < height; y++)
        {
            for (int x = 0; x < width; x++)
            {
                var p = image[x, y];
                float r = p.R / 255f;
                float g = p.G / 255f;
                float b = p.B / 255f;

                tensor[0, 0, y, x] = (r - Mean[0]) / Std[0]; // R
                tensor[0, 1, y, x] = (g - Mean[1]) / Std[1]; // G
                tensor[0, 2, y, x] = (b - Mean[2]) / Std[2]; // B
            }
        }

        return tensor;
    }

    private static DenseTensor<float> CreateInputTensorNHWC(Image<Rgb24> image)
    {
        int height = image.Height;
        int width = image.Width;

        var tensor = new DenseTensor<float>(new[] { 1, height, width, 3 });

        // NHWC layout: [1, H, W, C]
        for (int y = 0; y < height; y++)
        {
            for (int x = 0; x < width; x++)
            {
                var p = image[x, y];
                float r = p.R / 255f;
                float g = p.G / 255f;
                float b = p.B / 255f;

                tensor[0, y, x, 0] = (r - Mean[0]) / Std[0]; // R
                tensor[0, y, x, 1] = (g - Mean[1]) / Std[1]; // G
                tensor[0, y, x, 2] = (b - Mean[2]) / Std[2]; // B
            }
        }

        return tensor;
    }
}