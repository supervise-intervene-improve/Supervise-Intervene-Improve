using System.Collections;
using UnityEngine;

/// <summary>
/// Headset half of the end-to-end latency measurement. Reads the 64-bit sequence number
/// that the publisher drew into the corner of each RGB frame and echoes it back twice
/// over the EXISTING command socket (SessionCommandSender -> CmdListener):
///
///   phase 'A'  the instant the JPEG finished decoding
///   phase 'B'  after the frame carrying it has been presented (WaitForEndOfFrame)
///
/// Both echoes are timestamped on the PUBLISHER's clock when they arrive, so
/// (RTT_B - RTT_A) isolates the on-headset cost exactly, with no clock synchronisation
/// between the Quest and the PC. The Python reference (tools/latency_stamp.py) is optional and
/// not part of this release; without it the runtime disables the probe.
///
/// The decoder here MUST stay bit-identical to `read_stamp()` in tools/latency_stamp.py.
/// The five constants below are asserted against that reference by
/// CSharpParityTests.test_csharp_reader_matches_python -- if they drift, recovered
/// sequence numbers become garbage and nothing else in the pipeline would notice.
///
/// Off unless the publisher is stamping: an absent stamp fails the magic/checksum test
/// and is silently ignored, so this is inert on an unstamped session rather than wrong.
/// </summary>
public static class LatencyStampProbe
{
    // --- must match tools/latency_stamp.py exactly ---
    public const int BlockPx   = 8;   // one JPEG DCT block
    public const int StampBits = 64;
    public const int StampCols = 32;
    public const int Inset     = 2;   // sampled window inside each block
    private const ulong Magic  = 0x5A;

    public const int StampRows     = StampBits / StampCols;     // 2
    public const int RegionWidthPx = StampCols * BlockPx;       // 256
    public const int RegionHeightPx = StampRows * BlockPx;      // 16

    /// <summary>Set by InterventionSessionBootstrap once the session endpoint is known.</summary>
    public static bool   Enabled  = false;
    public static string EchoIp   = null;
    public static int    EchoPort = 0;

    // Diagnostics, surfaced in the 1 Hz panel heartbeat.
    public static long StampsRead;
    public static long StampsMissed;

    // Auto-dormancy. Reading the stamp costs two 256x16 region reads per decoded frame
    // (the second only when the first placement misses), which at 4 panels x ~56 Hz is
    // real work to spend on a session that is not being stamped at all. After
    // DormantAfterMisses consecutive misses with no hit, back off to sampling one frame
    // in DormantRetryEvery so the probe re-arms by itself the moment the publisher is
    // relaunched with --latency_probe. Leaving `Enabled` true is therefore safe.
    private const int DormantAfterMisses = 120;
    private const int DormantRetryEvery  = 300;
    private static int  _missStreak;
    private static bool _dormant;
    private static int  _dormantSkip;
    private static bool _dormantLogged;

    /// <summary>True while backing off because no stamped frames have been seen.</summary>
    public static bool IsDormant { get { return _dormant; } }

    // Sequence numbers decoded this frame, awaiting the end-of-frame echo. A tiny fixed
    // ring: at most one stamp per panel per frame, and a dropped entry is a lost sample,
    // never a stall.
    private const int PendingMax = 32;
    private static readonly uint[] _pending = new uint[PendingMax];
    private static int _pendingCount;
    private static readonly object _pendingLock = new object();

    /// <summary>
    /// Call immediately after a successful ImageConversion.LoadImage. Sends echo A and
    /// queues echo B for the end of the current frame.
    /// </summary>
    public static void OnDecoded(Texture2D tex, double decodeMs)
    {
        if (!Enabled || tex == null || string.IsNullOrEmpty(EchoIp) || EchoPort <= 0)
            return;

        if (_dormant)
        {
            // Cheap re-arm poll; skip the pixel read on the other frames.
            if (++_dormantSkip < DormantRetryEvery) return;
            _dormantSkip = 0;
        }

        uint seq;
        if (!TryRead(tex, out seq))
        {
            StampsMissed++;
            if (!_dormant && ++_missStreak >= DormantAfterMisses)
            {
                _dormant = true;
                if (!_dormantLogged)
                {
                    _dormantLogged = true;
                    Debug.Log("[LatencyStampProbe] no stamped frames seen in "
                              + DormantAfterMisses + " decodes; going dormant. Will re-arm "
                              + "automatically if the publisher starts stamping.");
                }
            }
            return;
        }
        _missStreak = 0;
        if (_dormant)
        {
            _dormant = false;
            Debug.Log("[LatencyStampProbe] stamped frames detected; probe re-armed.");
        }
        StampsRead++;

        SessionCommandSender.Send(EchoIp, EchoPort, Format("A", seq, decodeMs));

        lock (_pendingLock)
        {
            if (_pendingCount < PendingMax)
                _pending[_pendingCount++] = seq;
        }
    }

    /// <summary>Flushes the phase-B echoes. Driven by LatencyStampProbeDriver.</summary>
    public static void FlushPresented()
    {
        if (!Enabled || string.IsNullOrEmpty(EchoIp) || EchoPort <= 0)
            return;

        int count;
        uint[] snapshot;
        lock (_pendingLock)
        {
            count = _pendingCount;
            if (count == 0) return;
            snapshot = new uint[count];
            System.Array.Copy(_pending, snapshot, count);
            _pendingCount = 0;
        }
        for (int i = 0; i < count; i++)
            SessionCommandSender.Send(EchoIp, EchoPort, Format("B", snapshot[i], -1.0));
    }

    private static string Format(string phase, uint seq, double decodeMs)
    {
        // Matches format_echo() in tools/latency_stamp.py: LAT|<phase>|<seq>|<decode_ms>
        return string.Format(
            System.Globalization.CultureInfo.InvariantCulture,
            "LAT|{0}|{1}|{2:F3}", phase, seq, decodeMs);
    }

    /// <summary>
    /// Recover the sequence number, or false if this frame carries no valid stamp.
    ///
    /// Tries the stamp region at the TOP of the image and, failing that, at the BOTTOM.
    /// Unity's texture origin convention relative to the publisher's row 0 is not worth
    /// asserting blind from the host side -- the magic byte plus checksum reject a wrong
    /// guess outright, so trying both is strictly safer than assuming either, and costs
    /// one extra 256x16 read on frames that have no stamp at all.
    /// </summary>
    public static bool TryRead(Texture2D tex, out uint seq)
    {
        seq = 0;
        if (tex == null || tex.width < RegionWidthPx || tex.height < RegionHeightPx)
            return false;

        // GetPixels' origin is bottom-left. "Top of the image" is therefore the highest y.
        if (TryReadRegion(tex, tex.height - RegionHeightPx, true, out seq)) return true;
        if (TryReadRegion(tex, 0, false, out seq)) return true;
        return false;
    }

    private static bool TryReadRegion(Texture2D tex, int yOrigin, bool flipRows, out uint seq)
    {
        seq = 0;
        Color[] px;
        try
        {
            px = tex.GetPixels(0, yOrigin, RegionWidthPx, RegionHeightPx);
        }
        catch (UnityEngine.UnityException)
        {
            // Texture not readable (marked non-readable on load). Not recoverable here;
            // the caller counts it as a miss.
            return false;
        }
        if (px == null || px.Length < RegionWidthPx * RegionHeightPx)
            return false;

        ulong value = 0UL;
        for (int bit = 0; bit < StampBits; bit++)
        {
            int stampRow = bit / StampCols;
            int stampCol = bit % StampCols;

            double sum = 0.0;
            int n = 0;
            for (int dy = Inset; dy < BlockPx - Inset; dy++)
            {
                // Publisher-space pixel row of this sample, then mapped into the array.
                int ny = stampRow * BlockPx + dy;
                int r  = flipRows ? (RegionHeightPx - 1 - ny) : ny;
                int rowBase = r * RegionWidthPx;
                for (int dx = Inset; dx < BlockPx - Inset; dx++)
                {
                    Color c = px[rowBase + stampCol * BlockPx + dx];
                    // Match numpy's mean over the RGB channels of a uint8 image.
                    sum += (c.r + c.g + c.b) * (255.0 / 3.0);
                    n++;
                }
            }
            if (n == 0) return false;
            if (sum / n > 127.5) value |= (1UL << bit);
        }

        if (((value >> 48) & 0xFF) != Magic) return false;

        ulong checksum = 0UL;
        for (int i = 0; i < 7; i++) checksum ^= (value >> (8 * i)) & 0xFF;
        if (checksum != ((value >> 56) & 0xFF)) return false;

        seq = (uint)(value & 0xFFFFFFFF);
        return true;
    }
}

/// <summary>
/// Drives <see cref="LatencyStampProbe.FlushPresented"/> once per frame, after rendering.
///
/// WaitForEndOfFrame is the latest in-app hook there is; the remaining submit-to-photon
/// hop is what the Meta runtime reports as `Prd=` on its own VrApi line, which
/// tools/quest_telemetry.py already collects. So the two together cover the last leg
/// without this component having to guess at it.
/// </summary>
[DefaultExecutionOrder(10000)]
public class LatencyStampProbeDriver : MonoBehaviour
{
    private static LatencyStampProbeDriver _instance;

    public static void EnsureExists()
    {
        if (_instance != null) return;
        var go = new GameObject("LatencyStampProbeDriver");
        Object.DontDestroyOnLoad(go);
        _instance = go.AddComponent<LatencyStampProbeDriver>();
    }

    private void OnEnable()  { StartCoroutine(EndOfFrameLoop()); }

    private IEnumerator EndOfFrameLoop()
    {
        var wait = new WaitForEndOfFrame();
        while (true)
        {
            yield return wait;
            // Never let an echo failure take down the render loop.
            try { LatencyStampProbe.FlushPresented(); }
            catch (System.Exception ex)
            {
                Debug.LogWarning("[LatencyStampProbe] flush failed: " + ex.Message);
            }
        }
    }

    private void OnDestroy() { if (_instance == this) _instance = null; }
}
