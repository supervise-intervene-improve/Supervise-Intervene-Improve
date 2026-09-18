using System;
using System.Globalization;
using System.Net;
using System.Net.Sockets;
using System.Text;
using System.Threading;
using NetMQ;
using NetMQ.Sockets;
using UnityEngine;

/// <summary>
/// Binds a ZMQ PUB socket (Quest-side) and streams right controller pose + buttons at
/// <see cref="sendHz"/> Hz on topic <see cref="zmqTopic"/>. Python subscribes with a
/// plain SUB socket connecting to the Quest's IP on <see cref="zmqPort"/>.
///
/// Also broadcasts a UDP discovery beacon every <see cref="broadcastIntervalSeconds"/> s
/// so Python can auto-detect the Quest IP without hardcoding.
///
/// Coordinate convention: Unity → ROS (pos: forward→x, right→-y, up→z).
///
/// Lifecycle: starts DISABLED by default in SIIScene_InterventionV1; enabled/disabled by
/// MotionControllerModeManager. Always starts ENABLED in SIIScene_MotionControllerOnly.
///
/// NetMQ teardown: plain background Thread (no NetMQPoller). On disable/destroy, sets
/// _running = false, joins thread with 500 ms timeout, then calls
/// NetMQConfig.Cleanup(false) — the false flag avoids terminating the shared context used
/// by other subscribers in the same scene.
/// </summary>
public class MotionControllerZmqPublisher : MonoBehaviour
{
    [Header("ZMQ Publisher")]
    public int zmqPort = 6090;
    public string zmqTopic = "MotionController";
    // 90 Hz: pose is sampled in Update() (frame-rate bound ~72-90Hz on Quest), so sending faster
    // only re-ships duplicate stale frames — wasted radio/CPU on the headset.
    public int sendHz = 90;

    [Header("Yaw Reference")]
    // Compensate the Quest's arbitrary world-frame yaw (depends on headset boot/recenter
    // direction). At every OnEnable (= every intervention) we capture the head's horizontal
    // facing and publish poses in that yaw-aligned frame, so "push away from you" always maps
    // to the same robot axis regardless of where the app was booted.
    public bool yawCompensate = true;

    [Header("UDP Discovery Broadcast")]
    public int discoveryPort = 6656;
    public string deviceName = "MetaQuest3_RightController";
    public float broadcastIntervalSeconds = 0.75f;

    [Header("Coordinate Conversion")]
    public bool convertUnityToRos = true;

    [Header("Debug")]
    public bool printDebugJson = false;
    public bool printBroadcast = false;

    /// <summary>True once the ZMQ PUB socket binds successfully.</summary>
    public bool IsPublishing { get; private set; }

    private Thread _zmqThread;
    private Thread _broadcastThread;
    private volatile bool _running;

    private readonly object _jsonLock = new object();
    private string _latestJson = "";
    private long _latestSampleId;          // bumped per Update(); ZMQ thread sends each sample once
    private Quaternion _invYawRef = Quaternion.identity;

    // ---------------------------------------------------------------- data types

    [Serializable]
    private class RightControllerData
    {
        public HandData right;
        public bool A;
        public bool B;
        public bool X;
        public bool Y;
        public double unity_time;
        public long timestamp_ns;
    }

    [Serializable]
    private class HandData
    {
        public float[] pos;
        public float[] rot;   // xyzw
        public float[] vel;
        public float[] ang_vel;
        public float index_trigger;
        public float hand_trigger;
        public bool thumbstick_click;
        public float[] thumbstick;
    }

    [Serializable]
    private class DiscoveryMessage
    {
        public string type;
        public string device_name;
        public int zmq_port;
        public string topic;
        public long timestamp_ms;
    }

    // ---------------------------------------------------------------- lifecycle

    void OnEnable()
    {
        CultureInfo.CurrentCulture = CultureInfo.InvariantCulture;

        // Flush the previous session's pose so the ZMQ thread can never ship a stale frame
        // from before this enable (was a 1-frame jump hazard on 2nd interventions).
        lock (_jsonLock) { _latestJson = ""; _latestSampleId = 0; }

        CaptureYawReference();

        IsPublishing = false;
        _running = true;

        _zmqThread = new Thread(ZmqPublisherLoop) { IsBackground = true, Name = "MC_ZmqPub" };
        _zmqThread.Start();

        _broadcastThread = new Thread(UdpBroadcastLoop) { IsBackground = true, Name = "MC_UdpDisc" };
        _broadcastThread.Start();

        Debug.Log($"[MCPublisher] Enabled — ZMQ PUB tcp://*:{zmqPort}, UDP discovery :{discoveryPort}");
    }

    /// <summary>Capture the head's horizontal facing as the pose reference frame.</summary>
    private void CaptureYawReference()
    {
        _invYawRef = Quaternion.identity;
        if (!yawCompensate) return;

        var cam = Camera.main;
        Transform head = cam != null ? cam.transform : null;
        if (head == null) { Debug.LogWarning("[MCPublisher] No main camera — yaw compensation off."); return; }

        Vector3 fwd = head.forward;
        fwd.y = 0f;
        if (fwd.sqrMagnitude < 1e-6f) { Debug.LogWarning("[MCPublisher] Head looking straight up/down — yaw compensation off."); return; }

        _invYawRef = Quaternion.Inverse(Quaternion.LookRotation(fwd.normalized, Vector3.up));
        Debug.Log($"[MCPublisher] Yaw reference captured (head fwd={fwd.normalized}).");
    }

    void OnDisable() => Shutdown();
    void OnDestroy() => Shutdown();

    // ---------------------------------------------------------------- main-thread data collection

    void Update()
    {
        RightControllerData data = BuildData();
        string json = JsonUtility.ToJson(data);
        lock (_jsonLock) { _latestJson = json; _latestSampleId++; }
        if (printDebugJson) Debug.Log($"[MCPublisher] {json}");
    }

    // ---------------------------------------------------------------- ZMQ publisher thread

    private void ZmqPublisherLoop()
    {
        AsyncIO.ForceDotNet.Force();

        int sleepMs = Math.Max(1, (int)(1000.0f / Math.Max(1, sendHz)));

        try
        {
            using (var pub = new PublisherSocket())
            {
                pub.Options.SendHighWatermark = 1;
                pub.Bind($"tcp://*:{zmqPort}");
                IsPublishing = true;
                Debug.Log($"[MCPublisher] PUB socket bound tcp://*:{zmqPort}");

                long lastSentId = 0;
                while (_running)
                {
                    string json;
                    long sampleId;
                    lock (_jsonLock) { json = _latestJson; sampleId = _latestSampleId; }

                    // Send each main-thread sample exactly once — re-sending the same frame
                    // just ships duplicate stale packets (wasted radio + consumer drain work).
                    if (!string.IsNullOrEmpty(json) && sampleId != lastSentId)
                    {
                        try
                        {
                            pub.SendMoreFrame(zmqTopic).SendFrame(json);
                            lastSentId = sampleId;
                        }
                        catch (Exception e) { Debug.LogWarning($"[MCPublisher] Send failed: {e.Message}"); }
                    }
                    Thread.Sleep(sleepMs);
                }
            }
        }
        catch (Exception e)
        {
            Debug.LogWarning($"[MCPublisher] ZMQ thread error: {e.Message}");
        }
        finally
        {
            IsPublishing = false;
        }
    }

    // ---------------------------------------------------------------- UDP discovery thread

    private void UdpBroadcastLoop()
    {
        int sleepMs = Math.Max(100, (int)(broadcastIntervalSeconds * 1000f));
        var endpoint = new IPEndPoint(IPAddress.Broadcast, discoveryPort);

        using (var udp = new UdpClient())
        {
            udp.EnableBroadcast = true;

            while (_running)
            {
                try
                {
                    var msg = new DiscoveryMessage
                    {
                        type = "MQ3_ZMQ_DISCOVERY",
                        device_name = deviceName,
                        zmq_port = zmqPort,
                        topic = zmqTopic,
                        timestamp_ms = DateTimeOffset.UtcNow.ToUnixTimeMilliseconds()
                    };
                    string json = JsonUtility.ToJson(msg);
                    byte[] bytes = Encoding.UTF8.GetBytes(json);
                    udp.Send(bytes, bytes.Length, endpoint);
                    if (printBroadcast) Debug.Log($"[MCPublisher] UDP broadcast: {json}");
                }
                catch (Exception e) { Debug.LogWarning($"[MCPublisher] UDP broadcast failed: {e.Message}"); }

                Thread.Sleep(sleepMs);
            }
        }
    }

    // ---------------------------------------------------------------- data building

    private RightControllerData BuildData()
    {
        var data = new RightControllerData();
        data.right = BuildHandData();
        data.A = OVRInput.Get(OVRInput.RawButton.A);
        data.B = OVRInput.Get(OVRInput.RawButton.B);
        data.X = OVRInput.Get(OVRInput.RawButton.X);
        data.Y = OVRInput.Get(OVRInput.RawButton.Y);
        data.unity_time = Time.timeAsDouble;
        data.timestamp_ns = DateTimeOffset.UtcNow.ToUnixTimeMilliseconds() * 1_000_000L;
        return data;
    }

    private HandData BuildHandData()
    {
        var hand = new HandData();

        // Prefer the serialized anchor transform; fall back to OVR local pose.
        Vector3 pos;
        Quaternion rot;
        Transform anchor = FindRightAnchor();
        if (anchor != null)
        {
            pos = anchor.position;
            rot = anchor.rotation;
        }
        else
        {
            pos = OVRInput.GetLocalControllerPosition(OVRInput.Controller.RTouch);
            rot = OVRInput.GetLocalControllerRotation(OVRInput.Controller.RTouch);
        }

        Vector3 vel    = OVRInput.GetLocalControllerVelocity(OVRInput.Controller.RTouch);
        Vector3 angVel = OVRInput.GetLocalControllerAngularVelocity(OVRInput.Controller.RTouch);

        // Re-express the pose in the head-yaw frame captured at OnEnable, so the mapping is
        // independent of the Quest's arbitrary world-frame boot orientation. Constant
        // translation offsets cancel in the consumer's clutch-offset engagement, so rotating
        // (not origin-shifting) is sufficient.
        if (yawCompensate)
        {
            pos    = _invYawRef * pos;
            rot    = _invYawRef * rot;
            vel    = _invYawRef * vel;
            angVel = _invYawRef * angVel;
        }

        if (convertUnityToRos)
        {
            pos    = UnityPosToRos(pos);
            rot    = UnityRotToRos(rot);
            vel    = UnityPosToRos(vel);
            angVel = UnityPosToRos(angVel);
        }

        hand.pos     = new[] { pos.x,  pos.y,  pos.z };
        hand.rot     = new[] { rot.x,  rot.y,  rot.z, rot.w };
        hand.vel     = new[] { vel.x,  vel.y,  vel.z };
        hand.ang_vel = new[] { angVel.x, angVel.y, angVel.z };

        hand.index_trigger   = OVRInput.Get(OVRInput.RawAxis1D.RIndexTrigger);
        hand.hand_trigger    = OVRInput.Get(OVRInput.RawAxis1D.RHandTrigger);
        Vector2 stick        = OVRInput.Get(OVRInput.RawAxis2D.RThumbstick);
        hand.thumbstick      = new[] { stick.x, stick.y };
        hand.thumbstick_click = OVRInput.Get(OVRInput.RawButton.RThumbstick);

        return hand;
    }

    // ---------------------------------------------------------------- coordinate conversion

    private static Vector3 UnityPosToRos(Vector3 p) => new Vector3(p.z, -p.x, p.y);

    private static Quaternion UnityRotToRos(Quaternion q)
    {
        var m = Matrix4x4.Rotate(q);
        Vector3 rosUp  = UnityPosToRos(m.MultiplyVector(Vector3.up));
        Vector3 rosFwd = UnityPosToRos(m.MultiplyVector(Vector3.forward));
        return Quaternion.LookRotation(rosFwd, rosUp);
    }

    // ---------------------------------------------------------------- anchor lookup

    private Transform _cachedAnchor;
    private bool _anchorSearched;

    private Transform FindRightAnchor()
    {
        if (_cachedAnchor != null) return _cachedAnchor;
        if (_anchorSearched) return null;
        _anchorSearched = true;

        // Try the exact name shown in the Inspector screenshot.
        var go = GameObject.Find("MetaQuestTouchPlus_Right");
        if (go == null) go = GameObject.Find("RightHandAnchor");
        if (go != null)
        {
            _cachedAnchor = go.transform;
            Debug.Log($"[MCPublisher] Using anchor: {go.name}");
        }
        else
        {
            Debug.LogWarning("[MCPublisher] MetaQuestTouchPlus_Right / RightHandAnchor not found — using OVRInput local pose.");
        }
        return _cachedAnchor;
    }

    // ---------------------------------------------------------------- teardown

    private void Shutdown()
    {
        if (!_running) return;
        _running = false;

        _zmqThread?.Join(500);
        _broadcastThread?.Join(500);

        // Drop the last pose so the next enable can never replay it.
        lock (_jsonLock) { _latestJson = ""; _latestSampleId = 0; }

        // Do NOT call NetMQConfig.Cleanup here. The bool arg is `block`, NOT "keep the shared
        // context" — Cleanup() terminates the process-wide NetMQ context, which aborts every
        // other poller in the scene (risk chart, point clouds, status HUD) with a
        // TerminatingException under IL2CPP (observed crash on MC toggle). Our own PUB socket
        // is disposed by the `using` block in ZmqPublisherLoop once _running=false, so no
        // global cleanup is needed. Scene-exit teardown is owned by InterventionBackButton.
        IsPublishing = false;
        Debug.Log("[MCPublisher] Stopped.");
    }
}
