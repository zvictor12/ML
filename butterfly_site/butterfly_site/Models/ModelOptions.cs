namespace ButterflySite.Models;

public sealed class ModelOptions
{
    public const string SectionName = "Model";

    public string OnnxPath { get; init; } = "models/butterfly.onnx";
    public int EmbeddingSize { get; init; } = 1280;
}
