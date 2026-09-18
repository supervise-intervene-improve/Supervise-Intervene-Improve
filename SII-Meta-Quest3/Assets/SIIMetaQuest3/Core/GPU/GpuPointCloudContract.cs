using System;
using System.Buffers.Binary;
using UnityEngine;

public readonly struct GpuPcFrameHeader
{
    public readonly int DeclaredCapacity;
    public readonly int ActualCount;

    public GpuPcFrameHeader(int declaredCapacity, int actualCount)
    {
        DeclaredCapacity = declaredCapacity;
        ActualCount = actualCount;
    }
}

public static class GpuPointCloudContract
{
    public const uint Magic = 0x31435047; // "GPC1"
    public const int HeaderSize = 12;
    public const int PositionBytesPerPoint = 6; // int16 x 3
    public const int ColorBytesPerPoint = 3;    // uint8 x 3
    public const int BytesPerPoint = PositionBytesPerPoint + ColorBytesPerPoint;
    private const float MmToMeters = 0.001f;

    public static bool TryParseHeader(byte[] payload, out GpuPcFrameHeader header, out string error)
    {
        if (payload == null)
        {
            header = default;
            error = "Payload is null.";
            return false;
        }

        return TryParseHeader(payload.AsSpan(), out header, out error);
    }

    public static bool TryParseHeader(ReadOnlySpan<byte> payload, out GpuPcFrameHeader header, out string error)
    {
        header = default;
        error = null;

        if (payload.Length < HeaderSize)
        {
            error = $"Payload too small for GPU point-cloud header: {payload.Length} bytes.";
            return false;
        }

        uint magic = BinaryPrimitives.ReadUInt32LittleEndian(payload.Slice(0, 4));
        if (magic != Magic)
        {
            error = $"Unexpected GPU point-cloud magic 0x{magic:X8}.";
            return false;
        }

        int declaredCapacity = BinaryPrimitives.ReadInt32LittleEndian(payload.Slice(4, 4));
        int actualCount = BinaryPrimitives.ReadInt32LittleEndian(payload.Slice(8, 4));
        if (declaredCapacity < 0 || actualCount < 0)
        {
            error = $"Negative capacity/count in GPU point-cloud header: capacity={declaredCapacity} count={actualCount}.";
            return false;
        }
        if (actualCount > declaredCapacity)
        {
            error = $"GPU point-cloud count {actualCount} exceeds declared capacity {declaredCapacity}.";
            return false;
        }

        int expectedLength = HeaderSize + (actualCount * BytesPerPoint);
        if (payload.Length != expectedLength)
        {
            error = $"GPU point-cloud payload length mismatch: got {payload.Length}, expected {expectedLength}.";
            return false;
        }

        header = new GpuPcFrameHeader(declaredCapacity, actualCount);
        return true;
    }

    public static bool TryDecodeInto(
        byte[] payload,
        in GpuPcFrameHeader header,
        Vector3[] rawPoints,
        Color32[] colors,
        float positionScale,
        bool forceDebugColor,
        Color debugColor,
        out string error)
    {
        if (payload == null)
        {
            error = "Payload is null.";
            return false;
        }

        return TryDecodeInto(payload.AsSpan(), header, rawPoints, colors, positionScale,
            forceDebugColor, debugColor, out error);
    }

    public static bool TryDecodeInto(
        ReadOnlySpan<byte> payload,
        in GpuPcFrameHeader header,
        Vector3[] rawPoints,
        Color32[] colors,
        float positionScale,
        bool forceDebugColor,
        Color debugColor,
        out string error)
    {
        error = null;

        if (rawPoints == null || rawPoints.Length < header.DeclaredCapacity)
        {
            error = "Raw point buffer is missing or smaller than the declared capacity.";
            return false;
        }
        if (colors == null || colors.Length < header.DeclaredCapacity)
        {
            error = "Color buffer is missing or smaller than the declared capacity.";
            return false;
        }

        if (header.ActualCount == 0)
            return true;

        ReadOnlySpan<byte> body = payload.Slice(HeaderSize);
        ReadOnlySpan<byte> xyzBytes = body.Slice(0, header.ActualCount * PositionBytesPerPoint);
        ReadOnlySpan<byte> rgbBytes = body.Slice(header.ActualCount * PositionBytesPerPoint);

        float scale = positionScale * MmToMeters;
        Color32 debug = (Color32)debugColor;
        for (int i = 0; i < header.ActualCount; i++)
        {
            int xyzOffset = i * PositionBytesPerPoint;
            short xMm = BinaryPrimitives.ReadInt16LittleEndian(xyzBytes.Slice(xyzOffset + 0, 2));
            short yMm = BinaryPrimitives.ReadInt16LittleEndian(xyzBytes.Slice(xyzOffset + 2, 2));
            short zMm = BinaryPrimitives.ReadInt16LittleEndian(xyzBytes.Slice(xyzOffset + 4, 2));
            rawPoints[i] = new Vector3(xMm * scale, yMm * scale, zMm * scale);

            if (forceDebugColor)
            {
                colors[i] = debug;
            }
            else
            {
                int rgbOffset = i * ColorBytesPerPoint;
                colors[i] = new Color32(
                    rgbBytes[rgbOffset + 0],
                    rgbBytes[rgbOffset + 1],
                    rgbBytes[rgbOffset + 2],
                    255
                );
            }
        }

        return true;
    }
}
