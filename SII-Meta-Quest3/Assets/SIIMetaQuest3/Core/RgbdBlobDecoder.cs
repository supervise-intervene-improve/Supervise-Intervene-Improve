using System;
using System.Text;
using UnityEngine;

public static class RgbdBlobDecoder
{
    public struct Decoded
    {
        public string camName;
        public ushort width;
        public ushort height;
        public float fovyDeg;
        public byte[] rgbJpeg;
        public float[] depth;     // optional
        public float[] pointsXYZ; // optional (x,y,z,x,y,z...)
    }

    public static bool TryDecode(byte[] blob, out Decoded d)
    {
        d = default;
        if (blob == null || blob.Length < 4) return false;

        int off = 0;

        // MAGIC "SIIB"
        if (!(blob[0] == (byte)'S' && blob[1] == (byte)'I' && blob[2] == (byte)'I' && blob[3] == (byte)'B'))
            return false;
        off += 4;

        // u32 version (little endian)
        if (off + 4 > blob.Length) return false;
        uint version = BitConverter.ToUInt32(blob, off);
        off += 4;

        // u16 name length
        if (off + 2 > blob.Length) return false;
        ushort nameLen = BitConverter.ToUInt16(blob, off);
        off += 2;

        if (off + nameLen > blob.Length) return false;
        d.camName = Encoding.UTF8.GetString(blob, off, nameLen);
        off += nameLen;

        // header: <HHfIII (2+2+4+4+4+4 = 20 bytes)
        if (off + 20 > blob.Length) return false;

        d.width = BitConverter.ToUInt16(blob, off); off += 2;
        d.height = BitConverter.ToUInt16(blob, off); off += 2;
        d.fovyDeg = BitConverter.ToSingle(blob, off); off += 4;

        uint rgbLen = BitConverter.ToUInt32(blob, off); off += 4;
        uint depthLen = BitConverter.ToUInt32(blob, off); off += 4;
        uint pcPoints = BitConverter.ToUInt32(blob, off); off += 4;

        if (off + rgbLen + depthLen > blob.Length) return false;

        // RGB JPEG
        d.rgbJpeg = new byte[rgbLen];
        Buffer.BlockCopy(blob, off, d.rgbJpeg, 0, (int)rgbLen);
        off += (int)rgbLen;

        // Depth float32 bytes (depthLen should be width*height*4 if enabled)
        if (depthLen > 0)
        {
            int floatCount = (int)depthLen / 4;
            d.depth = new float[floatCount];
            Buffer.BlockCopy(blob, off, d.depth, 0, (int)depthLen);
            off += (int)depthLen;
        }

        // Pointcloud float32 xyz (pcPoints * 3 floats)
        int pcFloatCount = (int)pcPoints * 3;
        int pcBytes = pcFloatCount * 4;

        if (pcBytes > 0 && off + pcBytes <= blob.Length)
        {
            d.pointsXYZ = new float[pcFloatCount];
            Buffer.BlockCopy(blob, off, d.pointsXYZ, 0, pcBytes);
            off += pcBytes;
        }

        return true;
    }
}
