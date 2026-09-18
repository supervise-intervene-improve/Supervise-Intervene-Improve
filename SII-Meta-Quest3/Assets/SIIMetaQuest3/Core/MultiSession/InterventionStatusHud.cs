using System;
using System.Threading;
using NetMQ;
using NetMQ.Sockets;
using UnityEngine;
using UnityEngine.Rendering;

/// <summary>
/// In SIIScene_InterventionV1, subscribes to the selected session's status topic
/// (SimPub/Status/intervention) and shows a short head-relative HUD message when the
/// real robot releases into human control ("intervention_started" → "Intervention has
/// begun"). Python publishes the event at the exact release moment (see
/// runtime_impl.py queue_status), re-sent for a few frames for PUB/SUB reliability.
///
/// NetMQ teardown follows the project-wide safety pattern: OnMsg is guarded against a
/// terminated/disposed context, and OnDisable/OnDestroy stop the poller (blocking) before
/// disposing the socket. InterventionBackButton.StopSceneSubscribers disables it before
/// scene exit so its poller stops while the shared context is still alive.
/// </summary>
public class InterventionStatusHud : MonoBehaviour
{
    [Header("Topic")]
    public string statusTopic = "SimPub/Status/intervention";

    [Header("HUD")]
    public string startedMessage = "Intervention has begun";
    public string finishedMessage = "Intervention over";
    [Tooltip("Shown when the operator requested an intervention that could not start — "
           + "telekinesis needs the real arm, so an unreachable robot silently did nothing "
           + "before this existed.")]
    public string failedMessage = "Intervention unavailable - robot offline";
    public string busyMessage = "Intervention unavailable - robot in use";
    public string transitionFailedMessage = "Intervention unavailable - transition failed";
    public float displaySeconds = 3.0f;
    public string preferredAnchorName = "CenterEyeAnchor";
    public Vector3 localPosition = new Vector3(0f, 0.05f, 0.85f);
    public Color textColor = new Color(0.2f, 1.0f, 0.4f, 1.0f);
    public Color finishedColor = new Color(1.0f, 0.85f, 0.2f, 1.0f);
    public Color failedColor = new Color(1.0f, 0.35f, 0.3f, 1.0f);
    public int fontSize = 46;
    public float characterSize = 0.008f;

    /// <summary>True between an "intervention_started" and the next "intervention_finished"
    /// status event. Read by MotionControllerModeManager to gate MC mode to interventions only.</summary>
    public static bool InterventionActive { get; private set; }

    private string _ip;
    private int _topicPort;

    private SubscriberSocket _sub;
    private NetMQPoller _poller;
    private volatile bool _shuttingDown;
    // Event code set by poller thread, consumed on main thread: 1 = started, 2 = finished.
    private int _eventPending;
    private string _failureReason = string.Empty;

    private Transform _anchor;
    private GameObject _labelObject;
    private TextMesh _textMesh;
    private float _hideAt = -1f;

    void Start()
    {
        SessionRegistry.Load();
        if (!SessionRegistry.HasSelection)
        {
            enabled = false;
            return;
        }
        _ip = SessionRegistry.SelectedPublisherIp;
        _topicPort = SessionRegistry.SelectedTopicPort;
        InterventionActive = false;
        StartNetMq();
    }

    void Update()
    {
        // Consume any event flagged by the poller thread (1 = started, 2 = finished,
        // 3 = requested but could not start).
        int evt = Interlocked.Exchange(ref _eventPending, 0);
        if (evt == 1)
        {
            InterventionActive = true;
            ShowWithColor(startedMessage, textColor);
        }
        else if (evt == 2)
        {
            InterventionActive = false;
            ShowWithColor(finishedMessage, finishedColor);
        }
        else if (evt == 3)
        {
            // A failed start never entered HUMAN_CONTROL, so the arm is untouched and MC
            // mode must stay off — InterventionActive is explicitly cleared, not left as-is.
            InterventionActive = false;
            ShowWithColor(FailureMessageForReason(_failureReason), failedColor);
        }

        if (_labelObject != null)
        {
            // Keep it parented to the head anchor and facing the user.
            if (_anchor == null || !_anchor.gameObject.activeInHierarchy)
                _anchor = ResolveAnchor();
            if (_anchor != null && _labelObject.transform.parent != _anchor)
                _labelObject.transform.SetParent(_anchor, false);
            _labelObject.transform.localPosition = localPosition;
            _labelObject.transform.localRotation = Quaternion.identity;

            if (_hideAt > 0f && Time.unscaledTime >= _hideAt)
            {
                _labelObject.SetActive(false);
                _hideAt = -1f;
            }
        }
    }

    private string FailureMessageForReason(string reason)
    {
        if (string.Equals(reason, "robot_busy", StringComparison.Ordinal))
            return busyMessage;
        if (string.Equals(reason, "robot_unreachable", StringComparison.Ordinal)
            || string.Equals(reason, "robot_features_unavailable", StringComparison.Ordinal))
            return failedMessage;
        return transitionFailedMessage;
    }

    /// <summary>
    /// Show a HUD message with a specific color and optional duration override.
    /// Called by MotionControllerModeManager for local (non-ZMQ) mode-switch feedback.
    /// </summary>
    public void ShowWithColor(string msg, Color color, float? durationOverride = null)
    {
        if (_anchor == null) _anchor = ResolveAnchor();
        if (_labelObject == null) CreateLabel();
        if (_textMesh != null)
        {
            _textMesh.color = color;
            var mr = _labelObject.GetComponent<MeshRenderer>();
            // Rebuild the material from the current font so the color takes effect.
            if (mr != null && _textMesh.font != null) mr.sharedMaterial = _textMesh.font.material;
        }
        ShowMessage(msg, durationOverride);
    }

    private void ShowMessage(string msg, float? durationOverride = null)
    {
        float dur = durationOverride.HasValue ? durationOverride.Value : displaySeconds;
        if (_anchor == null) _anchor = ResolveAnchor();
        if (_labelObject == null) CreateLabel();
        if (_textMesh != null) _textMesh.text = msg;
        if (_labelObject != null) _labelObject.SetActive(true);
        _hideAt = Time.unscaledTime + Mathf.Max(0.5f, dur);
        Debug.Log($"[InterventionStatusHud] showing '{msg}' for {dur:F1}s");
    }

    private Transform ResolveAnchor()
    {
        var named = GameObject.Find(preferredAnchorName);
        if (named != null) return named.transform;
        var cam = Camera.main;
        return cam != null ? cam.transform : null;
    }

    private void CreateLabel()
    {
        if (_anchor == null) return;
        _labelObject = new GameObject("InterventionStatusHud_Label");
        _labelObject.transform.SetParent(_anchor, false);

        _textMesh = _labelObject.AddComponent<TextMesh>();
        var font = Resources.GetBuiltinResource<Font>("Arial.ttf");
        if (font != null) _textMesh.font = font;
        _textMesh.text = startedMessage;
        _textMesh.fontSize = fontSize;
        _textMesh.characterSize = characterSize;
        _textMesh.anchor = TextAnchor.MiddleCenter;
        _textMesh.alignment = TextAlignment.Center;
        _textMesh.fontStyle = FontStyle.Bold;
        _textMesh.color = textColor;

        var mr = _labelObject.GetComponent<MeshRenderer>();
        if (mr != null)
        {
            if (font != null) mr.sharedMaterial = font.material;
            mr.shadowCastingMode = ShadowCastingMode.Off;
            mr.receiveShadows = false;
            mr.lightProbeUsage = LightProbeUsage.Off;
            mr.reflectionProbeUsage = ReflectionProbeUsage.Off;
        }
        _labelObject.SetActive(false);
    }

    // ------------------------------------------------------------------ NetMQ

    /// <summary>Re-point the status subscription to a different session in place (combined-view
    /// switch, no scene reload). Reconnects the SimPub/Status/intervention SUB to the new
    /// endpoint so the "Intervention has begun" HUD tracks the newly-selected session.</summary>
    public void Reconfigure(string ip, int topicPort)
    {
        _ip = ip;
        _topicPort = topicPort;
        InterventionActive = false;
        Shutdown();       // Stop() the poller (blocking) before disposing the socket
        StartNetMq();     // resets _shuttingDown and reconnects on the new endpoint
        Debug.Log($"[InterventionStatusHud] reconfigured → tcp://{_ip}:{_topicPort}");
    }

    private void StartNetMq()
    {
        try
        {
            _shuttingDown = false;
            _sub = new SubscriberSocket();
            _sub.Options.ReceiveHighWatermark = 4;
            _sub.Options.Linger = TimeSpan.Zero;
            _sub.Connect($"tcp://{_ip}:{_topicPort}");
            _sub.Subscribe(statusTopic);
            _sub.ReceiveReady += OnMsg;
            _poller = new NetMQPoller { _sub };
            _poller.RunAsync();
            Debug.Log($"[InterventionStatusHud] subscribed tcp://{_ip}:{_topicPort} topic={statusTopic}");
        }
        catch (Exception e)
        {
            Debug.LogWarning($"[InterventionStatusHud] NetMQ start failed: {e.Message}");
        }
    }

    private void OnMsg(object sender, NetMQSocketEventArgs e)
    {
        if (_shuttingDown) return;

        var msg = new NetMQMessage();
        // Guard against the shared NetMQ context being terminated during scene unload
        // (unhandled here -> poller-thread abort under IL2CPP).
        bool got;
        try { got = e.Socket.TryReceiveMultipartMessage(ref msg); }
        catch (TerminatingException) { _shuttingDown = true; try { e.Socket.ReceiveReady -= OnMsg; } catch { } return; }
        catch (ObjectDisposedException) { _shuttingDown = true; try { e.Socket.ReceiveReady -= OnMsg; } catch { } return; }
        if (!got || msg.FrameCount < 1) return;

        string evt;
        try { evt = msg[msg.FrameCount - 1].ConvertToString(); }
        catch { return; }

        if (string.IsNullOrEmpty(evt)) return;
        if (evt.Contains("intervention_started"))
            Interlocked.Exchange(ref _eventPending, 1);
        else if (evt.Contains("intervention_finished"))
            Interlocked.Exchange(ref _eventPending, 2);
        // "intervention_failed" shares no substring with the two above, so ordering here is
        // not load-bearing — but keep it last so a future "..._finished_x" cannot shadow it.
        else if (evt.Contains("intervention_failed"))
        {
            int separator = evt.IndexOf(':');
            _failureReason = separator >= 0 && separator + 1 < evt.Length
                ? evt.Substring(separator + 1)
                : string.Empty;
            Interlocked.Exchange(ref _eventPending, 3);
        }
    }

    void OnDisable() => Shutdown();
    void OnDestroy() => Shutdown();

    private void Shutdown()
    {
        InterventionActive = false;
        if (_shuttingDown && _sub == null && _poller == null) return;
        _shuttingDown = true;

        var sub = _sub;
        var poller = _poller;
        _sub = null;
        _poller = null;

        try { if (sub != null) sub.ReceiveReady -= OnMsg; } catch { }
        // Stop() blocks until the poll loop exits so the socket is disposed only after the
        // poller stops iterating it (avoids "Must not be disposed" / TerminatingException aborts).
        try { if (poller != null && poller.IsRunning) poller.Stop(); } catch { }
        try { sub?.Close(); } catch { }
        try { sub?.Dispose(); } catch { }
        try { poller?.Dispose(); } catch { }

        if (_labelObject != null) { Destroy(_labelObject); _labelObject = null; _textMesh = null; }
    }
}
