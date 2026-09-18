using System;
using System.Text;
using UnityEngine;

[Serializable]
public class RgbdHeaderLite
{
    public string cam_name;
    public int width;
    public int height;
    public double timestamp;

    public int rgb_len;
    public int depth_len;
    public int pc_len;
}

[RequireComponent(typeof(Renderer))]
public class SimPubRgbdQuad : MonoBehaviour
{
    public SimPubClient client;
    public string topic = "SimPub/Sensors/top/rgbd";

    [Header("Options")]
    public bool flipY = true;

    [Header("Debug")]
    public bool log = true;
    public int logEveryNFrames = 60;

    private Texture2D _tex;
    private Renderer _r;
    private int _frame;

    private void Awake()
    {
        _r = GetComponent<Renderer>();

        _tex = new Texture2D(2, 2, TextureFormat.RGB24, false);

        var m = _r.material;
        if (m.HasProperty("_MainTex")) m.SetTexture("_MainTex", _tex);
        if (m.HasProperty("_BaseMap")) m.SetTexture("_BaseMap", _tex);
        m.mainTexture = _tex;

        _r.material.mainTextureScale = flipY ? new Vector2(1, -1) : Vector2.one;
    }

    private void Update()
    {
        _frame++;

        if (client == null) return;
        if (!client.TryGetLatest(topic, out var blob) || blob == null || blob.Length < 8) return;

        int jsonLen = BitConverter.ToInt32(blob, 0);
        if (jsonLen <= 0 || jsonLen > blob.Length - 4) return;

        string json = Encoding.UTF8.GetString(blob, 4, jsonLen);
        var hdr = JsonUtility.FromJson<RgbdHeaderLite>(json);
        if (hdr == null || string.IsNullOrEmpty(hdr.cam_name)) return;

        int offset = 4 + jsonLen;
        if (hdr.rgb_len <= 0 || offset + hdr.rgb_len > blob.Length) return;

        if (log && (_frame % logEveryNFrames == 0))
        {
            byte b0 = blob[offset + 0];
            byte b1 = blob[offset + 1];
            Debug.Log($"[SimPubRgbdQuad] topic={topic} cam={hdr.cam_name} " +
                      $"jsonLen={jsonLen} rgb_len={hdr.rgb_len} w={hdr.width} h={hdr.height} " +
                      $"jpegMagic={b0:X2} {b1:X2} blobBytes={blob.Length}");
        }

        var rgbJpg = new byte[hdr.rgb_len];
        Buffer.BlockCopy(blob, offset, rgbJpg, 0, hdr.rgb_len);

        bool ok = _tex.LoadImage(rgbJpg, markNonReadable: false);
        if (!ok) return;

        var m2 = _r.material;
        if (m2.HasProperty("_MainTex")) m2.SetTexture("_MainTex", _tex);
        if (m2.HasProperty("_BaseMap")) m2.SetTexture("_BaseMap", _tex);
        m2.mainTexture = _tex;
    }
}
