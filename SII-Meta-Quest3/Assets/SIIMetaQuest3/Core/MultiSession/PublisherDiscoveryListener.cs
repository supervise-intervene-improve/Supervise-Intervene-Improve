using System;
using System.Collections.Concurrent;
using System.Net;
using System.Net.Sockets;
using System.Text;
using System.Threading;
using UnityEngine;

/// <summary>
/// Listens for the launcher's UDP discovery beacon so the publisher PC IP is NOT
/// hardcoded in the scene. The launcher (multi_session_launcher.py) broadcasts a
/// small JSON beacon ~once/second on <see cref="discoveryPort"/>; this component
/// receives it and exposes the advertised IP + port scheme.
///
/// MultiSessionGridManager reads <see cref="DiscoveredIp"/> (and optionally the
/// ports) before spawning its grid, falling back to the serialized scene value if
/// no beacon arrives within its timeout.
///
/// Android note: receiving UDP broadcast on the Quest requires holding a Wi-Fi
/// MulticastLock — without it the OS silently drops broadcast/multicast frames to
/// save power. We acquire one in StartListening and release it on stop.
/// </summary>
public class PublisherDiscoveryListener : MonoBehaviour
{
    [Tooltip("UDP port the launcher broadcasts on. Must match the launcher's "
           + "--discovery_port (default 8720; not SimPub's 7720 multicast port).")]
    public int discoveryPort = 8720;

    [Tooltip("Beacon app tag — only beacons with this tag are accepted. Must match "
           + "the launcher's payload (\"SII_MULTI\").")]
    public string appTag = "SII_MULTI";

    /// <summary>Most recently advertised publisher IP, or null/empty until a beacon arrives.</summary>
    public string DiscoveredIp { get; private set; }
    public int    DiscoveredBaseTopicPort { get; private set; } = -1;
    public int    DiscoveredPortStep      { get; private set; } = -1;
    public int    DiscoveredSessions      { get; private set; } = -1;
    public int    DiscoveredGridCapacity  { get; private set; } = -1;
    /// <summary>True when the launcher advertised --rgb_mode: the selected session's
    /// single view should render the diamond RGB camera panels instead of point clouds.
    /// Propagated to SessionRegistry on selection (see MultiSessionGridManager).</summary>
    public bool   DiscoveredRgbMode       { get; private set; }
    /// <summary>True when the launcher advertised --mc_active: the Quest right controller
    /// is the intervention input device (motion controller). While an intervention is live
    /// the right controller must therefore be reserved for MC alone — see
    /// MotionControllerModeManager.McControlsLocked. Absent from an older launcher's
    /// beacon, which JsonUtility maps to false, i.e. "not MC" — the safe direction, since
    /// that only leaves the pre-existing controls unblocked.</summary>
    public bool   DiscoveredMcActive      { get; private set; }
    /// <summary>"quest" | "desktop" | "none" from the beacon; empty when the launcher
    /// predates the field, which leaves the Quest OOD supervisor disabled.</summary>
    public string DiscoveredOodOwner      { get; private set; } = "";
    // Explicit OOD opt-in from the launcher. Absent beacon field => false => OOD off,
    // which is the safe default for an older launcher.
    public bool   DiscoveredOodEnabled    { get; private set; }
    public float  LastRxTime              { get; private set; } = -1f;
    public bool   HasDiscovered => !string.IsNullOrEmpty(DiscoveredIp);

    // Static cache of the last good beacon. Statics survive scene reloads, so a fresh
    // listener created on a warm return to the selector can seed itself immediately
    // (HasDiscovered = true) instead of blocking for the next ~1/sec beacon. The live
    // UDP listener still runs and refreshes; MultiSessionGridManager.HandleDiscoveryIpChange
    // respawns the grid only if the IP/port layout actually changes.
    private static string s_lastIp;
    private static int    s_lastBaseTopicPort = -1;
    private static int    s_lastPortStep      = -1;
    private static int    s_lastSessions      = -1;
    private static int    s_lastGridCapacity  = -1;
    private static bool   s_lastRgbMode;
    private static bool   s_lastMcActive;
    private static string s_lastOodOwner = "";
    private static bool   s_lastOodEnabled;

    // Public accessors to the static cache — used by RiskBarChart and any other component
    // that needs session topology (IP, port scheme, count) in scenes beyond the selector.
    public static string LastIp            => s_lastIp;
    public static int    LastBaseTopicPort => s_lastBaseTopicPort;
    public static int    LastPortStep      => s_lastPortStep;
    public static int    LastNSessions     => s_lastSessions;
    public static int    LastGridCapacity  => s_lastGridCapacity;
    public static bool   LastRgbMode       => s_lastRgbMode;
    /// <summary>Static mirror of DiscoveredMcActive. Static because the right-controller
    /// lock has to be answerable from SIIScene_InterventionV1, which has no listener
    /// instance of its own — the beacon is received by the selector scene.</summary>
    public static bool   LastMcActive      => s_lastMcActive;
    public static string LastOodOwner      => s_lastOodOwner;
    public static bool   LastOodEnabled    => s_lastOodEnabled;
    public static bool   HasCachedDiscovery => !string.IsNullOrEmpty(s_lastIp);

    private UdpClient _udp;
    private Thread    _thread;
    private volatile bool _running;
    private readonly ConcurrentQueue<string> _msgQueue = new ConcurrentQueue<string>();

#if UNITY_ANDROID && !UNITY_EDITOR
    private AndroidJavaObject _multicastLock;
#endif

    void OnEnable()  => StartListening();
    void OnDisable() => StopListening();
    void OnDestroy() => StopListening();

    void Update()
    {
        // Parse beacons on the main thread (Time.* and property writes).
        while (_msgQueue.TryDequeue(out var json))
            ParseBeacon(json);
    }

    private void StartListening()
    {
        if (_running) return;

        // Seed from the static cache so a warm re-entry has HasDiscovered = true immediately,
        // letting the grid spawn without waiting for the next beacon. The RxLoop below still
        // refreshes these from live beacons.
        if (string.IsNullOrEmpty(DiscoveredIp) && !string.IsNullOrEmpty(s_lastIp))
        {
            DiscoveredIp            = s_lastIp;
            DiscoveredBaseTopicPort = s_lastBaseTopicPort;
            DiscoveredPortStep      = s_lastPortStep;
            DiscoveredSessions      = s_lastSessions;
            DiscoveredGridCapacity  = s_lastGridCapacity;
            DiscoveredRgbMode       = s_lastRgbMode;
            DiscoveredMcActive      = s_lastMcActive;
            DiscoveredOodOwner      = s_lastOodOwner;
            DiscoveredOodEnabled    = s_lastOodEnabled;
            Debug.Log($"[PublisherDiscovery] Seeded from cache: ip={s_lastIp} "
                    + $"baseTopic={s_lastBaseTopicPort} step={s_lastPortStep} n={s_lastSessions} rgbMode={s_lastRgbMode}");
        }

        AcquireMulticastLock();
        try
        {
            _udp = new UdpClient();
            _udp.Client.SetSocketOption(SocketOptionLevel.Socket, SocketOptionName.ReuseAddress, true);
            _udp.EnableBroadcast = true;
            _udp.Client.Bind(new IPEndPoint(IPAddress.Any, discoveryPort));

            _running = true;
            _thread = new Thread(RxLoop) { IsBackground = true, Name = "PubDiscovery" };
            _thread.Start();
            Debug.Log($"[PublisherDiscovery] Listening for '{appTag}' beacons on UDP {discoveryPort}.");
        }
        catch (Exception e)
        {
            Debug.LogWarning($"[PublisherDiscovery] Failed to start listener on UDP {discoveryPort}: {e.Message}");
            _running = false;
        }
    }

    private void RxLoop()
    {
        var remote = new IPEndPoint(IPAddress.Any, 0);
        while (_running)
        {
            try
            {
                byte[] data = _udp.Receive(ref remote);   // blocks
                if (data != null && data.Length > 0)
                    _msgQueue.Enqueue(Encoding.UTF8.GetString(data));
            }
            catch (SocketException)        { if (!_running) break; }
            catch (ObjectDisposedException){ break; }
            catch                          { /* ignore malformed datagrams */ }
        }
    }

    private void ParseBeacon(string json)
    {
        try
        {
            var b = JsonUtility.FromJson<Beacon>(json);
            if (b == null || b.app != appTag || string.IsNullOrEmpty(b.ip)) return;

            bool changed = b.ip != DiscoveredIp;
            DiscoveredIp            = b.ip;
            DiscoveredBaseTopicPort = b.base_topic_port;
            DiscoveredPortStep      = b.port_step;
            DiscoveredSessions      = b.n_sessions;
            DiscoveredGridCapacity  = b.grid_capacity;
            DiscoveredRgbMode       = b.rgb_mode;
            DiscoveredMcActive      = b.mc_active;
            DiscoveredOodOwner      = b.ood_owner ?? "";
            DiscoveredOodEnabled    = b.ood_enabled;
            LastRxTime              = Time.unscaledTime;

            // Update the static cache so a future warm re-entry can seed itself instantly.
            s_lastIp            = b.ip;
            s_lastBaseTopicPort = b.base_topic_port;
            s_lastPortStep      = b.port_step;
            s_lastSessions      = b.n_sessions;
            s_lastGridCapacity  = b.grid_capacity;
            s_lastRgbMode       = b.rgb_mode;
            s_lastMcActive      = b.mc_active;
            s_lastOodOwner      = b.ood_owner ?? "";
            s_lastOodEnabled    = b.ood_enabled;

            if (changed)
                Debug.Log($"[PublisherDiscovery] Beacon: ip={b.ip} baseTopic={b.base_topic_port} "
                        + $"step={b.port_step} n={b.n_sessions} rgbMode={b.rgb_mode} "
                        + $"mcActive={b.mc_active} gridCapacity={b.grid_capacity}");
        }
        catch { /* not a beacon we understand */ }
    }

    [Serializable]
    private class Beacon
    {
        public string app;
        public string ip;
        public int    base_topic_port;
        public int    port_step;
        public int    n_sessions;
        /// <summary>Visible selector slots. Zero/absent means n_sessions for backwards compatibility.</summary>
        public int    grid_capacity;
        public bool   rgb_mode;
        /// <summary>True when the launcher ran with MC_ACTIVE=1 (Quest right controller
        /// drives the arm during an intervention). Absent on older launchers => false.</summary>
        public bool   mc_active;
        /// <summary>"quest" | "desktop" | "none" — which side arbitrates OOD auto-pause.
        /// Absent from older launchers, in which case JsonUtility leaves it null and the
        /// Quest supervisor stays off (the safe direction: no phantom pauses).</summary>
        public string ood_owner;
        public bool   ood_enabled;
    }

    private void AcquireMulticastLock()
    {
#if UNITY_ANDROID && !UNITY_EDITOR
        try
        {
            using (var up = new AndroidJavaClass("com.unity3d.player.UnityPlayer"))
            {
                var activity = up.GetStatic<AndroidJavaObject>("currentActivity");
                var wifi     = activity.Call<AndroidJavaObject>("getSystemService", "wifi");
                _multicastLock = wifi.Call<AndroidJavaObject>("createMulticastLock", "SII_PubDiscovery");
                _multicastLock.Call("setReferenceCounted", true);
                _multicastLock.Call("acquire");
                Debug.Log("[PublisherDiscovery] Wi-Fi MulticastLock acquired.");
            }
        }
        catch (Exception e)
        {
            Debug.LogWarning($"[PublisherDiscovery] MulticastLock acquire failed (broadcast RX may not work): {e.Message}");
        }
#endif
    }

    private void ReleaseMulticastLock()
    {
#if UNITY_ANDROID && !UNITY_EDITOR
        try { _multicastLock?.Call("release"); } catch { }
        _multicastLock = null;
#endif
    }

    private void StopListening()
    {
        _running = false;
        try { _udp?.Close(); } catch { }
        try { _thread?.Join(200); } catch { }
        _udp    = null;
        _thread = null;
        ReleaseMulticastLock();
    }
}
