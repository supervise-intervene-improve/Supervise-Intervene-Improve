using System;
using System.Collections.Concurrent;
using System.Threading;
using NetMQ;
using NetMQ.Sockets;
using UnityEngine;
using UnityEngine.UI;

/// <summary>
/// A single panel in the 5x3 grid.  Subscribes to one MuJoCo session's top-camera
/// RGB feed, decodes the JPEG on the main thread, and displays it on a RawImage.
/// Responds to hover/select events from MultiSessionGridManager.
/// </summary>
[RequireComponent(typeof(Collider))]
public class SessionThumbnailPanel : MonoBehaviour
{
    [Header("Session Config")]
    public int    sessionIndex  = 0;
    public int    topicPort     = 7741;
    public int    commandPort   = 7746;  // topicPort + 5 — Python --cmd_port for A/B/X/Y forwarding
    public string publisherIp   = "127.0.0.1";
    public string sessionLabel  = "Session 0";
    [Tooltip("Visible disabled placeholder only: no subscriber, command socket, risk, or selection.")]
    public bool inactive = false;
    public bool IsInactive => inactive;
    public string thumbnailTopic = "SimPub/Sensors/front/rgb";
    /// <summary>True when the launcher advertised --rgb_mode (set by MultiSessionGridManager
    /// from the discovery beacon). Passed through to SessionRegistry on Select() so
    /// InterventionSessionBootstrap knows whether to render the diamond RGB panels
    /// instead of point clouds.</summary>
    public bool   rgbMode = false;

    [Header("Display")]
    public RawImage thumbnailImage;
    public Text     labelText;

    [Header("Visibility")]
    [Tooltip("Thumbnail quad renderer. When placeholders are enabled it is visible "
           + "before the first frame, but selection stays disabled until live data arrives.")]
    public Renderer thumbnailRenderer;
    [Tooltip("Root collider used for ray selection — disabled until the first frame "
           + "so empty placeholders cannot be hovered or selected.")]
    public Collider panelCollider;
    public bool showPlaceholderUntilFrame = true;
    public Color placeholderColor = new Color(0.45f, 0.45f, 0.50f, 1f);
    [Tooltip("Whether this panel can be ray-hovered/selected. Set false for the single-view RGB " +
             "diamond camera panels: in combined view BOTH scenes are loaded, so the grid's " +
             "right-controller ray would otherwise hit these colliders and Select() them — " +
             "writing a bogus session into SessionRegistry and breaking scene switching. " +
             "When false the collider stays disabled permanently (the feed still reveals at full " +
             "brightness — this gates the collider only, not the thumbnail visuals).")]
    public bool enableSelection = true;

    [Header("Risk Coloring")]
    [Tooltip("Set false for panels that are not session-grid panels (e.g. RGB diamond camera panels " +
             "in single view) so they don't get risk-based border coloring.")]
    public bool enableRiskColoring = true;

    [Header("Live Indicator")]
    [Tooltip("Set false for panels that are not session-grid panels (e.g. RGB diamond camera panels " +
             "in single view) so they don't show the LIVE indicator.")]
    public bool enableLiveIndicator = true;
    [Tooltip("Red dot + \"LIVE\" text shown only when this panel's session matches the " +
             "currently active single-view session (combined view). Hidden by default.")]
    public GameObject liveIndicatorRoot;

    [Header("Paused Indicator")]
    [Tooltip("Set false for panels that are not session-grid panels (e.g. RGB diamond camera " +
             "panels in single view) so they don't show the PAUSED badge.")]
    public bool enablePausedIndicator = true;
    [Tooltip("Amber \"❚❚ PAUSED\" badge shown when this panel's session reports sim-paused on " +
             "SimPub/Status/paused. Hidden by default. Positioned separately from LIVE so a " +
             "session can show both at once.")]
    public GameObject pausedIndicatorRoot;
    [Tooltip("Optional TextMesh inside pausedIndicatorRoot. Used to append an OOD pause reason.")]
    public TextMesh pausedIndicatorText;

    [Header("Hover / Select Visuals")]
    public Renderer borderRenderer;
    public Color    normalBorderColor  = new Color(0.25f, 0.25f, 0.25f, 1f);
    public Color    hoverBorderColor   = new Color(0.0f,  0.75f, 1.0f,  1f);
    public Color    selectedBorderColor = new Color(0.0f, 1.0f,  0.3f,  1f);
    public float    hoverScaleMultiplier = 1.06f;

    [Header("Reconnect")]
    [Tooltip("How long with no frames before attempting reconnect. "
           + "Short enough to recover after selector↔single-view NetMQ teardown.")]
    public float reconnectIdleSeconds = 8.0f;

    // --- internal ---
    private Texture2D _lastTexture;
    /// <summary>Last decoded texture (may be null until first frame arrives).</summary>
    public Texture LastTexture => _lastTexture;

    // Risk value received from SimPub/Status/risk. Stored as int bits for thread-safe
    // Interlocked.Exchange; reinterpreted as float in Update().
    private int _riskRaw;

    // Paused flag received from SimPub/Status/paused (0/1). Thread-safe via Interlocked;
    // read in Update() to toggle the PAUSED badge.
    private int _pausedRaw;

    // Intervention flag received from SimPub/Status/intervening (0/1). A LEVEL, republished
    // at ~4 Hz by the runtime, so it self-heals if a message is lost. The OOD supervisor
    // uses this to protect ANY intervening session, not just the one in single view.
    private int _interveningRaw;
    private bool _oodPaused;

    private bool _revealed;
    /// <summary>True once the panel has received its first frame and become visible.
    /// Stays true for the rest of the session (reveal-on-first-frame is one-way).</summary>
    public bool IsRevealed => _revealed;
    public float RiskValue => System.BitConverter.ToSingle(
        System.BitConverter.GetBytes(Interlocked.CompareExchange(ref _riskRaw, 0, 0)), 0);
    public bool IsPaused => Interlocked.CompareExchange(ref _pausedRaw, 0, 0) == 1;
    public bool IsIntervening => Interlocked.CompareExchange(ref _interveningRaw, 0, 0) == 1;

    private SubscriberSocket _sub;
    private NetMQPoller      _poller;
    private volatile bool    _shuttingDown;

    private readonly ConcurrentQueue<byte[]> _jpegQueue = new ConcurrentQueue<byte[]>();
    private Texture2D _tex; // backing texture for LoadImage
    private bool      _isHovered;
    private bool      _isSelected;
    private Vector3   _baseScale;
    private float     _lastRxTime;
    private float     _nextReconnectAt;
    private int       _netMqGeneration;
    private long      _rxCount;

    // --- Telemetry heartbeat -------------------------------------------------
    // _rxCount was previously only printed on reconnect/shutdown, so the RGB panel
    // update rate could not be measured from logcat. This 1 Hz line mirrors the one
    // SimPubRgbdSubscriber already emits; tools/quest_telemetry.py differences the
    // cumulative counter to recover the on-device receive rate. Log line only — no HUD.
    [Header("Telemetry")]
    [Tooltip("Seconds between 'hb' log lines. 0 disables the heartbeat (the default): "
           + "15 grid panels each logging at 1 Hz is 15 log lines/second on device, which "
           + "is real overhead. Set to 1 only while collecting telemetry with "
           + "tools/quest_telemetry.py.")]
    public float heartbeatLogIntervalSeconds = 0.0f;
    private float  _nextHeartbeatAt;
    private double _decodeMsAccum;
    private int    _decodeCount;
    private long   _dropCount;

    [Header("Decode Budget")]
    [Tooltip("Maximum JPEG decodes per panel per second. Thumbnails do not need the "
           + "display refresh rate; 0 disables the per-panel limit.")]
    public float maxDecodeHz = 10.0f;
    [Tooltip("Maximum panels allowed to decode in a single frame across ALL panels. "
           + "Prevents 15+ synchronous JPEG decodes landing in one frame. 0 = unlimited.")]
    public int maxDecodesPerFrame = 3;
    private byte[] _pendingJpeg;
    private float  _lastDecodeTime;
    private long   _skippedDecodes;

    // Global (static) per-frame decode budget shared by every panel instance.
    private static int s_decodeBudgetFrame = -1;
    private static int s_decodesThisFrame;

    /// <summary>Per-panel rate limit plus a shared per-frame budget. Panels are served in
    /// whatever order Unity runs their Update(), and because a losing panel keeps its
    /// pending frame it will win a later frame — so no panel is starved.</summary>
    private bool DecodeAllowed()
    {
        if (maxDecodeHz > 0f && (Time.unscaledTime - _lastDecodeTime) < (1f / maxDecodeHz))
            return false;
        if (maxDecodesPerFrame <= 0)
            return true;
        if (s_decodeBudgetFrame != Time.frameCount)
        {
            s_decodeBudgetFrame = Time.frameCount;
            s_decodesThisFrame = 0;
        }
        if (s_decodesThisFrame >= maxDecodesPerFrame)
        {
            _skippedDecodes++;
            return false;
        }
        s_decodesThisFrame++;
        return true;
    }

    // -----------------------------------------------------------------------

    void Start()
    {
        AsyncIO.ForceDotNet.Force();
        _baseScale = transform.localScale;
        _lastRxTime = Time.unscaledTime;
        _nextReconnectAt = Time.unscaledTime + reconnectIdleSeconds;

        if (labelText != null)
            labelText.text = sessionLabel;
        if (pausedIndicatorText == null && pausedIndicatorRoot != null)
            pausedIndicatorText = pausedIndicatorRoot.GetComponentInChildren<TextMesh>(true);

        ApplyBorderColor(normalBorderColor);

        if (inactive)
        {
            // Study C inactive slots are intentionally presentation-only. Do not call
            // StartNetMq: the absence of a runtime worker must also mean no socket,
            // policy, risk stream, intervention command, or event source exists.
            _revealed = false;
            enableSelection = false;
            SetVisible(true, false);
            Debug.Log($"[SessionThumbnailPanel] S{sessionIndex} INACTIVE placeholder (no sockets).");
            return;
        }

        // Start as a visible placeholder but keep selection disabled. The panel
        // becomes selectable only after the session publishes its first frame.
        _revealed = false;
        SetVisible(showPlaceholderUntilFrame, false);

        StartNetMq();
    }

    /// <summary>Toggle the panel's visuals + selection collider. Hides visuals only —
    /// never SetActive(false)/enabled=false, which would trigger OnDisable→Shutdown
    /// and kill the ZMQ subscriber so a late-starting session could never be detected.</summary>
    private void SetVisible(bool visuals, bool selectable)
    {
        if (thumbnailRenderer != null) thumbnailRenderer.enabled = visuals;
        if (borderRenderer    != null) borderRenderer.enabled    = visuals;
        if (labelText         != null) labelText.enabled         = visuals;
        // Gate the collider on enableSelection too — non-selectable panels (single-view RGB
        // diamond) keep it disabled permanently so the grid ray can never hit/select them.
        if (panelCollider     != null) panelCollider.enabled     = selectable && enableSelection;
        ApplyThumbnailColor(selectable ? Color.white : placeholderColor);
    }

    private static Color RiskColor(float risk)
    {
        float t = Mathf.Clamp01(risk);
        return t < 0.5f
            ? Color.Lerp(Color.green,  Color.yellow, t * 2f)
            : Color.Lerp(Color.yellow, Color.red,    (t - 0.5f) * 2f);
    }

    void Update()
    {
        if (inactive) return;
        // Apply risk-based border color when not in hover/selected state.
        // Disabled for non-grid panels (RGB diamond camera panels) via enableRiskColoring.
        if (enableRiskColoring)
        {
            normalBorderColor = RiskColor(RiskValue);
            if (!_isHovered && !_isSelected) ApplyBorderColor(normalBorderColor);
        }

        // Drive both the LIVE indicator AND the "selected" (green + enlarged) visual from the
        // current single-view session, so they follow the active session and SELF-CLEAR on the
        // panel you switched away from. Nothing else ever calls SetSelected(false), so without
        // this the previous panel kept its green/enlarged look and (because RefreshVisuals gives
        // _isSelected priority over _isHovered) showed no hover feedback — looking "locked"
        // though it was still selectable. Gated on enableLiveIndicator so this only applies to
        // grid panels (diamond RGB panels default sessionIndex=0 and must never auto-select).
        // Only meaningful in combined view; in exclusive-swap the grid is unloaded during single
        // view so SelectedSessionIndex never matches a live grid panel.
        if (enableLiveIndicator)
        {
            bool isCurrent = SessionRegistry.HasSelection
                          && SessionRegistry.SelectedSessionIndex == sessionIndex;

            if (liveIndicatorRoot != null && liveIndicatorRoot.activeSelf != isCurrent)
            {
                liveIndicatorRoot.SetActive(isCurrent);
                Debug.Log($"[SessionThumbnailPanel] S{sessionIndex} LIVE indicator -> {(isCurrent ? "ON" : "OFF")} "
                        + $"(selectedIndex={SessionRegistry.SelectedSessionIndex})");
            }

            if (_isSelected != isCurrent)
                SetSelected(isCurrent);
        }

        // PAUSED badge — driven by the authoritative SimPub/Status/paused flag. Independent of
        // the LIVE indicator, so a session can show both at once (positioned separately).
        if (enablePausedIndicator && pausedIndicatorRoot != null)
        {
            bool paused = IsPaused;
            if (pausedIndicatorRoot.activeSelf != paused)
                pausedIndicatorRoot.SetActive(paused);
        }

        // Dequeue latest JPEG and upload to texture (main thread only). Intermediate
        // frames are intentionally dropped (latest-wins); count them so the heartbeat
        // can distinguish "publisher too fast" from "frames never arrived".
        byte[] latest = null;
        while (_jpegQueue.TryDequeue(out var frame))
        {
            if (latest != null) _dropCount++;
            latest = frame;
        }

        // Decode budget. Previously all 15 grid panels (plus 4 diamond panels in RGB mode)
        // called ImageConversion.LoadImage on the SAME main thread every frame they had
        // data — up to 19 synchronous JPEG decodes + texture uploads in one frame, with no
        // throttling. Two limits now apply:
        //   * a per-panel minimum interval, because a thumbnail does not need 72 Hz; and
        //   * a global per-frame budget, round-robined by panel index so the same panels
        //     are not always the ones that win.
        // A skipped frame is held in _pendingJpeg (newest wins) so a panel that loses the
        // budget race still shows the latest image on its next turn rather than stalling.
        if (latest != null)
        {
            _lastRxTime = Time.unscaledTime;
            _pendingJpeg = latest;
        }

        if (_pendingJpeg != null && DecodeAllowed())
        {
            latest = _pendingJpeg;
            _pendingJpeg = null;
            _lastDecodeTime = Time.unscaledTime;

            if (_tex == null)
                _tex = new Texture2D(2, 2, TextureFormat.RGB24, false);
            long _decodeT0 = System.Diagnostics.Stopwatch.GetTimestamp();
            bool _decoded = ImageConversion.LoadImage(_tex, latest, false);
            double _decodeMs =
                (System.Diagnostics.Stopwatch.GetTimestamp() - _decodeT0) * 1000.0
                / System.Diagnostics.Stopwatch.Frequency;
            _decodeMsAccum += _decodeMs;
            _decodeCount++;
            if (_decoded)
            {
                _lastTexture = _tex;
                if (thumbnailImage != null)
                    thumbnailImage.texture = _tex;

                // End-to-end latency probe. Inert unless the publisher is stamping AND
                // LatencyStampProbe has been pointed at a command endpoint; an unstamped
                // frame fails the magic/checksum test and is ignored. Wrapped because a
                // measurement must never be able to break the panel it measures.
                if (LatencyStampProbe.Enabled)
                {
                    try
                    {
                        // The pure LoadImage cost, not re-measured past the texture
                        // assignment -- the PC subtracts this from RTT_A to isolate the
                        // network term, so it has to be the decode and nothing else.
                        LatencyStampProbe.OnDecoded(_tex, _decodeMs);
                    }
                    catch (System.Exception ex)
                    {
                        Debug.LogWarning("[SessionThumbnailPanel] latency probe: " + ex.Message);
                    }
                }

                // Reveal-on-first-frame: this session is publishing, so enable
                // selection. One-way — once revealed it stays live for the session.
                if (!_revealed)
                {
                    _revealed = true;
                    SetVisible(true, true);
                }
            }
        }

        EmitHeartbeatIfDue();

        // Idle reconnect
        if (Time.unscaledTime > _nextReconnectAt &&
            Time.unscaledTime - _lastRxTime > reconnectIdleSeconds)
        {
            Reconnect();
            _nextReconnectAt = Time.unscaledTime + reconnectIdleSeconds;
        }
    }

    /// <summary>1 Hz telemetry line for adb logcat. Same shape as the heartbeats in
    /// SimPubRgbdSubscriber / GpuMergedPointCloudLoader so one parser handles all
    /// three. rx is cumulative — the collector differences it into a receive rate.</summary>
    private void EmitHeartbeatIfDue()
    {
        if (heartbeatLogIntervalSeconds <= 0f) return;
        float now = Time.unscaledTime;
        if (now < _nextHeartbeatAt)
        {
            if (_nextHeartbeatAt - now > heartbeatLogIntervalSeconds * 2f)
                _nextHeartbeatAt = now + heartbeatLogIntervalSeconds; // clock moved back
            return;
        }
        _nextHeartbeatAt = now + heartbeatLogIntervalSeconds;

        double decodeAvgMs = _decodeCount > 0 ? _decodeMsAccum / _decodeCount : 0.0;
        _decodeMsAccum = 0.0;
        _decodeCount = 0;

        Debug.Log(
            $"[SessionThumbnailPanel] S{sessionIndex:00} hb "
            + $"rx={Interlocked.Read(ref _rxCount)} "
            + $"drops={_dropCount} "
            + $"skipped={_skippedDecodes} "
            + $"decode_ms={decodeAvgMs:F2} "
            + $"idle_s={(now - _lastRxTime):F2} "
            + $"revealed={_revealed} "
            + $"topic={thumbnailTopic} "
            + $"endpoint=tcp://{publisherIp}:{topicPort}");
    }

    // OnDisable fires when enabled=false (called by MultiSessionGridManager
    // before scene load to shut down pollers cleanly).
    void OnDisable()  => Shutdown();
    void OnDestroy()  => Shutdown();

    // -----------------------------------------------------------------------
    // Public API called by MultiSessionGridManager

    public void SetHover(bool hovered)
    {
        if (_isHovered == hovered) return;
        _isHovered = hovered;
        RefreshVisuals();
    }

    public void SetSelected(bool selected)
    {
        _isSelected = selected;
        RefreshVisuals();
    }

    public void SetOodPauseReason(bool isOodPaused)
    {
        if (_oodPaused == isOodPaused) return;
        _oodPaused = isOodPaused;
        if (pausedIndicatorText != null)
            pausedIndicatorText.text = _oodPaused ? "❚❚ PAUSED (OOD)" : "❚❚ PAUSED";
    }

    /// <summary>Commit this session to SessionRegistry and signal scene switch.</summary>
    public void Select()
    {
        if (inactive) return;
        int tPort  = topicPort;
        int sPort  = tPort - 1;  // service port is topic_port - 1 by scheme

        SessionRegistry.Save(
            index:       sessionIndex,
            topicPort:   tPort,
            servicePort: sPort,
            publisherIp: publisherIp,
            label:       sessionLabel,
            rgbMode:     rgbMode
        );
        SetSelected(true);
    }

    /// <summary>Re-point this panel to a DIFFERENT session in place (combined-view session
    /// switch without a scene reload). Drops the stale image — re-hides to placeholder until
    /// the new session publishes its first frame — then reconnects the ZMQ subscriber to the
    /// new endpoint. The per-camera topic string is unchanged; only ip/port move.</summary>
    public void Reconfigure(string ip, int newTopicPort)
    {
        publisherIp = ip;
        topicPort   = newTopicPort;
        commandPort = newTopicPort + 5;   // topic_port + 5 by scheme
        _revealed = false;
        SetVisible(showPlaceholderUntilFrame, false); // placeholder until the new first frame
        Reconnect();                                  // Shutdown + StartNetMq on the new endpoint
        Debug.Log($"[SessionThumbnailPanel] reconfigured → tcp://{ip}:{newTopicPort} topic={thumbnailTopic}");
    }

    // -----------------------------------------------------------------------
    // Visuals

    private void RefreshVisuals()
    {
        if (_isSelected)
        {
            ApplyBorderColor(selectedBorderColor);
            transform.localScale = _baseScale * hoverScaleMultiplier;
        }
        else if (_isHovered)
        {
            ApplyBorderColor(hoverBorderColor);
            transform.localScale = _baseScale * hoverScaleMultiplier;
        }
        else
        {
            ApplyBorderColor(normalBorderColor);
            transform.localScale = _baseScale;
        }
    }

    // Reused across calls: ApplyBorderColor runs every frame from Update() (risk tint), so
    // allocating a MaterialPropertyBlock per call produced 15 garbage objects per frame
    // across the grid. A MaterialPropertyBlock is a plain container — reusing one is safe
    // as long as it is refilled before each SetPropertyBlock, which GetPropertyBlock does.
    private MaterialPropertyBlock _borderMpb;
    private MaterialPropertyBlock _thumbMpb;
    private Color _lastBorderColor = new Color(-1f, -1f, -1f, -1f);

    private void ApplyBorderColor(Color c)
    {
        if (borderRenderer == null) return;
        // Skip entirely when the colour has not changed — the common case once a panel
        // settles into a steady risk value.
        if (c == _lastBorderColor) return;
        _lastBorderColor = c;
        if (_borderMpb == null) _borderMpb = new MaterialPropertyBlock();
        borderRenderer.GetPropertyBlock(_borderMpb);
        // Set BOTH properties: Custom/UnlitDoubleSided reads "_Color", but a default URP
        // material (Android) reads "_BaseColor". Writing only "_Color" left the risk tint
        // invisible on-device — same fix as ApplyThumbnailColor / RiskBarChart.SetQuadColor.
        _borderMpb.SetColor("_Color", c);
        _borderMpb.SetColor("_BaseColor", c);
        borderRenderer.SetPropertyBlock(_borderMpb);
    }

    private void ApplyThumbnailColor(Color c)
    {
        if (thumbnailRenderer == null) return;
        if (_thumbMpb == null) _thumbMpb = new MaterialPropertyBlock();
        thumbnailRenderer.GetPropertyBlock(_thumbMpb);
        _thumbMpb.SetColor("_Color", c);
        _thumbMpb.SetColor("_BaseColor", c);
        thumbnailRenderer.SetPropertyBlock(_thumbMpb);
    }

    // -----------------------------------------------------------------------
    // NetMQ

    private void StartNetMq()
    {
        int generation = ++_netMqGeneration;
        Debug.Log(
            $"[SessionThumbnailPanel] S{sessionIndex} NetMQ start gen={generation} "
            + $"tcp://{publisherIp}:{topicPort} topic={thumbnailTopic}"
        );
        try
        {
            _shuttingDown = false;
            _sub = new SubscriberSocket();
            _sub.Options.ReceiveHighWatermark = 2;
            _sub.Options.Linger = TimeSpan.Zero;
            _sub.Connect($"tcp://{publisherIp}:{topicPort}");
            _sub.Subscribe(thumbnailTopic);
            _sub.Subscribe("SimPub/Status/risk");
            _sub.Subscribe("SimPub/Status/paused");
            _sub.Subscribe("SimPub/Status/intervening");

            _sub.ReceiveReady += OnMsg;
            _poller = new NetMQPoller { _sub };
            _poller.RunAsync();
            _nextReconnectAt = Time.unscaledTime + Mathf.Max(0.5f, reconnectIdleSeconds);

            Debug.Log($"[SessionThumbnailPanel] S{sessionIndex} connected gen={generation} tcp://{publisherIp}:{topicPort} topic={thumbnailTopic}");
        }
        catch (Exception e)
        {
            Debug.LogWarning($"[SessionThumbnailPanel] S{sessionIndex} NetMQ start failed gen={generation}: {e.Message}");
            _nextReconnectAt = Time.unscaledTime + 1.0f;
        }
    }

    private void OnMsg(object sender, NetMQSocketEventArgs e)
    {
        if (_shuttingDown) return;
        try
        {
            var msg = new NetMQMessage();
            if (!e.Socket.TryReceiveMultipartMessage(ref msg)) return;
            if (msg.FrameCount < 2) return;
            // Frame 0 = topic string, Frame 1 = payload
            string topic = msg[0].ConvertToString();
            if (topic.EndsWith("/rgb"))
            {
                _jpegQueue.Enqueue(msg[1].ToByteArray());
                Interlocked.Increment(ref _rxCount);
            }
            else if (topic == "SimPub/Status/risk")
            {
                if (float.TryParse(msg[1].ConvertToString(),
                    System.Globalization.NumberStyles.Float,
                    System.Globalization.CultureInfo.InvariantCulture, out float r))
                {
                    Interlocked.Exchange(ref _riskRaw,
                        System.BitConverter.ToInt32(System.BitConverter.GetBytes(r), 0));
                }
            }
            else if (topic == "SimPub/Status/paused")
            {
                Interlocked.Exchange(ref _pausedRaw,
                    msg[1].ConvertToString().Trim() == "1" ? 1 : 0);
            }
            else if (topic == "SimPub/Status/intervening")
            {
                Interlocked.Exchange(ref _interveningRaw,
                    msg[1].ConvertToString().Trim() == "1" ? 1 : 0);
            }
        }
        catch { /* ignore parse errors */ }
    }

    private void Reconnect()
    {
        Debug.LogWarning(
            $"[SessionThumbnailPanel] S{sessionIndex} reconnect after idle rx={Interlocked.Read(ref _rxCount)} "
            + $"tcp://{publisherIp}:{topicPort}"
        );
        Shutdown();
        _lastRxTime = Time.unscaledTime;
        _nextReconnectAt = Time.unscaledTime + Mathf.Max(0.5f, reconnectIdleSeconds);
        StartNetMq();
    }

    private void Shutdown()
    {
        if (_shuttingDown) return;
        _shuttingDown = true;

        var sub = _sub;
        var poller = _poller;
        _sub = null;
        _poller = null;

        Debug.Log(
            $"[SessionThumbnailPanel] S{sessionIndex} shutdown gen={_netMqGeneration} "
            + $"hadSub={sub != null} hadPoller={poller != null} pollerRunning={(poller != null && poller.IsRunning)} "
            + $"rx={Interlocked.Read(ref _rxCount)} tcp://{publisherIp}:{topicPort}"
        );
        try { if (sub != null) sub.ReceiveReady -= OnMsg; } catch { }
        // Stop() BLOCKS until the poll loop exits, so the socket below is disposed only
        // after the poller is no longer iterating it. StopAsync() returned immediately and
        // the running poller then called Remove() on the disposed socket → "Must not be
        // disposed" on the poller thread → IL2CPP process abort on scene exit/stop.
        try { if (poller != null && poller.IsRunning) poller.Stop(); } catch { }
        try { sub?.Close(); } catch { }
        try { sub?.Dispose(); } catch { }
        try { poller?.Dispose(); } catch { }
    }
}
