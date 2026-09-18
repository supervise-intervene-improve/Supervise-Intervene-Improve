using System.Collections.Generic;
using UnityEngine;

/// <summary>
/// Owns the runtime-managed "MujocoScene" GameObject that the point cloud
/// loader anchors to. Ports the QR-anchor scaffold from the QR-anchor blueprint:
///
///   - Inspector-exposed AlignmentData struct mirroring the QR blueprint's
///     QRSceneAlignmentData (pos/euler/fixZAxis lists). Not used by the
///     current build but defines the wire format so a future QR detector
///     slots in without changing downstream consumers.
///   - Coordinate conversion GetPos()/GetRot() copied verbatim from
///     (QR blueprint) Assets/SceneLoader/Scripts/QRSceneAlignment.cs:15-26
///     (MuJoCo X-fwd/Y-left/Z-up → Unity X-right/Y-up/Z-fwd).
///   - Controller-based pose tuning (port of the QR blueprint PointCloudLoader.cs
///     HandleControllers): left-menu = lock toggle; left-stick = translate;
///     left-trigger/grip = Z; A/B = rotate X; X/Y = rotate Y; stick-clicks
///     = rotate Z. Default LOCKED so it doesn't fight with replay buttons.
///   - PlayerPrefs persistence: saved pose restored at Start, re-saved on
///     every lock-re-engagement.
///   - Future hook: ApplyAlignmentData(AlignmentData) for QR backends.
///
/// Attached to the InterventionSessionBootstrap GameObject when entering
/// SIIScene_InterventionV1 from the multi-session selector.
/// </summary>
public class SceneAnchorManager : MonoBehaviour
{
    // -------- Wire-format mirror of the QR blueprint QRSceneAlignmentData --------
    [System.Serializable]
    public class AlignmentData
    {
        public string qrText;
        public List<float> pos;     // MuJoCo coords (X-fwd, Y-left, Z-up)
        public List<float> euler;   // MuJoCo euler (X, Y, Z) degrees
        public bool fixZAxis;       // true = keep gravity-aligned (use Yaw only)

        public Vector3 GetPos()
        {
            // From QRSceneAlignment.cs: Unity = (-pos[1], pos[2], pos[0])
            if (pos == null || pos.Count < 3) return Vector3.zero;
            return new Vector3(-pos[1], pos[2], pos[0]);
        }

        public Quaternion GetRot()
        {
            if (euler == null || euler.Count < 3) return Quaternion.identity;
            if (fixZAxis)
                return Quaternion.Euler(0f, euler[2], 0f);
            return Quaternion.Euler(euler[1], -euler[2], -euler[0]);
        }
    }

    [Header("Anchor")]
    public string anchorName = "MujocoScene";

    [Header("Default Placement (used only on fresh install)")]
    [Tooltip("X = right, Y = up, Z = forward (relative to camera at scene load).")]
    public Vector3 defaultLocalOffset = new Vector3(0f, -0.3f, 1.0f);

    [Header("Controller Tuning")]
    [Tooltip("Default LOCKED so the controller can't fight with replay buttons. "
           + "HOLD Left Menu for adjustToggleHoldSeconds to unlock for nudging, hold again to "
           + "relock + save.")]
    public bool controlsLocked = true;
    [Tooltip("Seconds Left Menu must be held CONTINUOUSLY to toggle adjust/alignment mode. A tap "
           + "does nothing. Guards against an accidental brush of the button unlocking the anchor "
           + "mid-session — which also silently suppresses every other command while unlocked. "
           + "Set to 0 to restore the old instant toggle.")]
    public float adjustToggleHoldSeconds = AdjustModeHoldGate.DefaultHoldSeconds;
    public float translationSpeed = 0.2f;       // m/s on each axis
    public float rotationSpeed = 30.0f;         // deg/s on each axis

    [Header("Scale")]
    [Tooltip("Uniform scale applied to the MujocoScene anchor. 1.0 = exact MuJoCo size. "
           + "Adjustable in-VR via right stick Y when unlocked (Left Menu).")]
    public float sceneScale = 1.08f;
    [Tooltip("Scale change rate per second when nudging with right stick Y (relative, e.g. 0.4 = ±40%/s).")]
    public float scaleSpeed = 0.4f;

    [Header("Persistence")]
    public string playerPrefsKeyPos = "SceneAnchor_Pos";
    public string playerPrefsKeyRot = "SceneAnchor_Rot";
    public string playerPrefsKeyScale = "SceneAnchor_Scale";
    public bool persistPoseAcrossSessions = true;

    /// <summary>
    /// True while the anchor is UNLOCKED (PC-adjust mode). Other components
    /// (InterventionButtonForwarder, InterventionBackButton) read this to suppress
    /// sim/intervention/exit/cancel so the adjust controls (A/B/X/Y rotation, sticks,
    /// L-triggers) don't double-fire. Static so any component can query it without a ref.
    /// </summary>
    public static bool AdjustModeActive { get; private set; }

    /// <summary>
    /// True while a SceneAnchorManager is alive, i.e. SIIScene_InterventionV1 is loaded and
    /// the Left Menu toggle above is reachable.
    ///
    /// This component only exists in the single-view scene (it is AddComponent'd by
    /// InterventionSessionBootstrap), and combined view UNLOADS that scene on exit to the
    /// selector — so in the pure selector nothing owns Left Menu and AdjustModeActive is
    /// permanently false. Anything that gates a control on adjust mode has to know the
    /// difference between "locked" and "there is no lock here to unlock", or it silently
    /// becomes unreachable in the selector. MultiSessionGridManager uses this to run its own
    /// equivalent toggle when it is the only scene loaded.
    ///
    /// A count, not a bool: during a combined-view session switch the incoming scene's
    /// instance can enable before the outgoing one disables, and a bool would be left false.
    /// </summary>
    public static bool Exists => s_instanceCount > 0;
    private static int s_instanceCount;

    private GameObject _anchor;
    private readonly AdjustModeHoldGate _lockHold = new AdjustModeHoldGate();
    private InterventionStatusHud _hud;
    private bool _hasUnsavedNudges;

    void OnEnable()
    {
        s_instanceCount++;
    }

    void Start()
    {
        // Create or reuse the named GameObject.
        //
        // In combined view this object OUTLIVES the component that made it: it is created with
        // `new GameObject`, which puts it in the ACTIVE scene (the selector), while this
        // component lives in the additively-loaded SIIScene_InterventionV1. Exiting single view
        // unloads that scene, so on the next entry a fresh SceneAnchorManager finds the SAME
        // anchor still sitting exactly where the operator left it.
        var existing = GameObject.Find(anchorName);
        bool reusedExistingAnchor = existing != null;
        if (reusedExistingAnchor)
        {
            _anchor = existing;
            Debug.Log($"[SceneAnchorManager] reusing existing '{anchorName}' at {_anchor.transform.position}.");
        }
        else
        {
            _anchor = new GameObject(anchorName);
        }

        // Place: PlayerPrefs (if saved) > keep a reused anchor's live pose > camera+forward.
        if (persistPoseAcrossSessions && TryLoadPose(out var savedPos, out var savedRot))
        {
            _anchor.transform.SetPositionAndRotation(savedPos, savedRot);
            Debug.Log($"[SceneAnchorManager] restored pose from PlayerPrefs: pos={savedPos} rot={savedRot.eulerAngles}");
        }
        else if (reusedExistingAnchor)
        {
            // No saved pose, but the anchor already exists and is where the scene currently is.
            // Re-placing it camera-relative here is what made the point cloud jump UP on every
            // re-entry: the fallback sits at roughly eye height (defaultLocalOffset), well above
            // a table, and until the operator has unlocked/nudged/RE-LOCKED at least once there
            // is no saved pose to prefer over it. Leave the live pose alone.
            Debug.Log($"[SceneAnchorManager] no saved pose — keeping the existing '{anchorName}' "
                    + $"pose pos={_anchor.transform.position} rot={_anchor.transform.rotation.eulerAngles} "
                    + "(NOT re-placing camera-relative).");
        }
        else
        {
            var cam = Camera.main;
            Vector3 pos;
            Quaternion rot = Quaternion.identity;
            if (cam != null)
            {
                pos = cam.transform.position
                    + cam.transform.forward * defaultLocalOffset.z
                    + cam.transform.up      * defaultLocalOffset.y
                    + cam.transform.right   * defaultLocalOffset.x;
                // Face away from the viewer so its +Z is "into the scene".
                rot = Quaternion.LookRotation(Vector3.ProjectOnPlane(cam.transform.forward, Vector3.up).normalized, Vector3.up);
            }
            else
            {
                pos = Vector3.zero;
                Debug.LogWarning("[SceneAnchorManager] Camera.main is null — placing anchor at world origin.");
            }
            _anchor.transform.SetPositionAndRotation(pos, rot);
            Debug.Log($"[SceneAnchorManager] spawned '{anchorName}' at default pos={pos} (camera-relative).");
        }

        // Restore scale from PlayerPrefs, or use inspector default.
        if (persistPoseAcrossSessions && PlayerPrefs.HasKey(playerPrefsKeyScale))
        {
            sceneScale = PlayerPrefs.GetFloat(playerPrefsKeyScale);
            Debug.Log($"[SceneAnchorManager] restored scale from PlayerPrefs: {sceneScale:F3}");
        }
        _anchor.transform.localScale = Vector3.one * sceneScale;
        Debug.Log($"[SceneAnchorManager] scale applied: {sceneScale:F3}");

        AdjustModeActive = !controlsLocked;
        Debug.Log($"[SceneAnchorManager] Locked = {controlsLocked} (Left Menu to toggle).");
    }

    /// <summary>Head-relative HUD feedback for the hold-to-toggle, reusing the existing
    /// InterventionStatusHud (ShowWithColor is purely visual — it does NOT touch
    /// InterventionActive, so this cannot disturb MC/OOD gating).
    ///
    /// The HUD disables itself when there is no session selection, and a disabled component's
    /// Update never runs — so a message pushed to it would never hide or follow the head. Hence
    /// the isActiveAndEnabled check: no HUD is better than a stuck one.</summary>
    private void ShowHudMessage(string msg, Color color, float seconds)
    {
        if (_hud == null || !_hud.isActiveAndEnabled)
            _hud = FindAnyObjectByType<InterventionStatusHud>();
        if (_hud != null && _hud.isActiveAndEnabled)
            _hud.ShowWithColor(msg, color, seconds);
    }

    void OnDisable()
    {
        s_instanceCount = Mathf.Max(0, s_instanceCount - 1);
        // Persist before disappearing. The ONLY other save site is the unlock->re-lock edge in
        // Update(), so an operator who nudged the anchor and then exited single view (which
        // UNLOADS this scene in combined view) without completing another Left Menu hold lost
        // the tuning silently -- and, with nothing saved, the next entry fell back to the
        // camera-relative default. The anchor GameObject itself survives in the selector scene,
        // so this is only about making the pose stick across an app restart.
        //
        // The hold-to-toggle gate makes this MORE load-bearing, not less: re-locking is now a
        // multi-second deliberate hold, so "nudged it, then left without re-locking" is a more
        // likely way to exit than it was with the instant tap.
        SavePendingNudges("component disabled / scene unloaded");
        _lockHold.Reset();
        // Reset the shared flag so the next scene/session starts in normal (non-adjust) control.
        AdjustModeActive = false;
    }

    /// <summary>Android suspends rather than quits, so OnApplicationPause is the last reliable
    /// point to flush PlayerPrefs before the headset sleeps or the app is backgrounded.</summary>
    void OnApplicationPause(bool paused)
    {
        if (paused) SavePendingNudges("application paused");
    }

    void OnApplicationQuit()
    {
        SavePendingNudges("application quitting");
    }

    private void SavePendingNudges(string reason)
    {
        if (!persistPoseAcrossSessions || !_hasUnsavedNudges || _anchor == null) return;
        SavePose(_anchor.transform.position, _anchor.transform.rotation);
        _hasUnsavedNudges = false;
        Debug.Log($"[SceneAnchorManager] flushed unsaved anchor nudges ({reason}).");
    }

    void Update()
    {
        if (_anchor == null) return;

#if !UNITY_EDITOR
        // ----- Lock toggle (HOLD Left Menu / Start button) -----
        // Hold, not tap: a stray brush of Left Menu used to flip adjust mode instantly, and since
        // AdjustModeActive suppresses ALL command forwarding, the operator's next intervention /
        // sim / exit presses would then do nothing with no obvious cause.
        // Ignore the toggle entirely while the risk chart is being placed (L3), so the two adjust
        // modes stay mutually exclusive — passing that into Poll cancels any in-progress hold.
        bool lockPressed = OVRInput.Get(OVRInput.RawButton.Start) && !RiskBarChart.AdjustModeActive;

        bool lockToggled = _lockHold.Poll(lockPressed, adjustToggleHoldSeconds);

        // Feedback the moment the button goes down: a multi-second hold with no response is
        // indistinguishable from a dead button. Message duration = the hold itself, so it
        // clears right as the toggle lands (or shortly after an early release).
        if (_lockHold.JustPressed && adjustToggleHoldSeconds > 0f)
        {
            ShowHudMessage(
                controlsLocked
                    ? $"Hold {adjustToggleHoldSeconds:F0}s to UNLOCK alignment…"
                    : $"Hold {adjustToggleHoldSeconds:F0}s to LOCK alignment…",
                Color.white,
                adjustToggleHoldSeconds);
        }

        if (lockToggled)
        {
            controlsLocked = !controlsLocked;
            AdjustModeActive = !controlsLocked;
            Debug.Log($"[SceneAnchorManager] Locked = {controlsLocked} (AdjustModeActive={AdjustModeActive}) "
                    + $"after {adjustToggleHoldSeconds:F1}s hold");
            // On RE-LOCK, persist the current pose.
            if (controlsLocked && _hasUnsavedNudges && persistPoseAcrossSessions)
            {
                SavePose(_anchor.transform.position, _anchor.transform.rotation);
                _hasUnsavedNudges = false;
            }
            ShowHudMessage(
                controlsLocked ? "Alignment LOCKED" : "Alignment UNLOCKED",
                controlsLocked ? Color.white : new Color(1f, 0.75f, 0f, 1f),
                2.0f);
        }

        if (controlsLocked) return;

        // ----- Translation -----
        Vector3 move = Vector3.zero;
        Vector2 leftStick = OVRInput.Get(OVRInput.RawAxis2D.LThumbstick);
        move.x += leftStick.x;
        move.y += leftStick.y;
        if (OVRInput.Get(OVRInput.RawButton.LIndexTrigger)) move.z += 1.0f;
        if (OVRInput.Get(OVRInput.RawButton.LHandTrigger))  move.z -= 1.0f;
        if (move.sqrMagnitude > 0f)
        {
            _anchor.transform.position += move * translationSpeed * Time.deltaTime;
            _hasUnsavedNudges = true;
        }

        // ----- Rotation -----
        float rotStep = rotationSpeed * Time.deltaTime;
        // Right A = +pitch, Right B = -pitch
        if (OVRInput.Get(OVRInput.RawButton.A))
            _anchor.transform.rotation = Quaternion.AngleAxis( rotStep, Vector3.right) * _anchor.transform.rotation;
        if (OVRInput.Get(OVRInput.RawButton.B))
            _anchor.transform.rotation = Quaternion.AngleAxis(-rotStep, Vector3.right) * _anchor.transform.rotation;
        // Left X = -yaw, Left Y = +yaw
        if (OVRInput.Get(OVRInput.RawButton.X))
            _anchor.transform.rotation = Quaternion.AngleAxis(-rotStep, Vector3.up) * _anchor.transform.rotation;
        if (OVRInput.Get(OVRInput.RawButton.Y))
            _anchor.transform.rotation = Quaternion.AngleAxis( rotStep, Vector3.up) * _anchor.transform.rotation;
        // Stick clicks = roll
        if (OVRInput.Get(OVRInput.RawButton.RThumbstick))
            _anchor.transform.rotation = Quaternion.AngleAxis( rotStep, Vector3.forward) * _anchor.transform.rotation;
        if (OVRInput.Get(OVRInput.RawButton.LThumbstick))
            _anchor.transform.rotation = Quaternion.AngleAxis(-rotStep, Vector3.forward) * _anchor.transform.rotation;

        if (OVRInput.Get(OVRInput.RawButton.A) || OVRInput.Get(OVRInput.RawButton.B)
            || OVRInput.Get(OVRInput.RawButton.X) || OVRInput.Get(OVRInput.RawButton.Y)
            || OVRInput.Get(OVRInput.RawButton.RThumbstick) || OVRInput.Get(OVRInput.RawButton.LThumbstick))
        {
            _hasUnsavedNudges = true;
        }

        // ----- Scale (right stick Y when unlocked) -----
        Vector2 rightStick = OVRInput.Get(OVRInput.RawAxis2D.RThumbstick);
        if (Mathf.Abs(rightStick.y) > 0.1f)
        {
            sceneScale *= 1.0f + rightStick.y * scaleSpeed * Time.deltaTime;
            sceneScale = Mathf.Clamp(sceneScale, 0.1f, 5.0f);
            _anchor.transform.localScale = Vector3.one * sceneScale;
            _hasUnsavedNudges = true;
        }
#endif
    }

    /// <summary>Future QR detector entry point.
    /// Converts the supplied (MuJoCo-frame) AlignmentData via GetPos/GetRot
    /// and snaps the MujocoScene transform to that pose.</summary>
    public void ApplyAlignmentData(AlignmentData data)
    {
        if (_anchor == null || data == null) return;
        _anchor.transform.SetPositionAndRotation(data.GetPos(), data.GetRot());
        if (persistPoseAcrossSessions)
            SavePose(_anchor.transform.position, _anchor.transform.rotation);
        Debug.Log($"[SceneAnchorManager] ApplyAlignmentData: pos={_anchor.transform.position} rot={_anchor.transform.rotation.eulerAngles}");
    }

    private void SavePose(Vector3 pos, Quaternion rot)
    {
        // PrefsPose.Format writes InvariantCulture. String interpolation would use the DEVICE
        // culture, and on a comma-decimal locale that produced six comma-separated tokens which
        // TryLoadPose then rejected -- so every restore silently fell back to camera-relative.
        PlayerPrefs.SetString(playerPrefsKeyPos, PrefsPose.Format(pos));
        var e = rot.eulerAngles;
        PlayerPrefs.SetString(playerPrefsKeyRot, PrefsPose.Format(e));
        PlayerPrefs.SetFloat(playerPrefsKeyScale, sceneScale);
        PlayerPrefs.Save();
        Debug.Log($"[SceneAnchorManager] saved pose to PlayerPrefs: pos={pos} euler={e} scale={sceneScale:F3}");
    }

    private bool TryLoadPose(out Vector3 pos, out Quaternion rot)
    {
        pos = Vector3.zero;
        rot = Quaternion.identity;
        if (!PlayerPrefs.HasKey(playerPrefsKeyPos) || !PlayerPrefs.HasKey(playerPrefsKeyRot))
            return false;
        try
        {
            if (!PrefsPose.TryParseVector3(PlayerPrefs.GetString(playerPrefsKeyPos), out pos))
            {
                Debug.LogWarning("[SceneAnchorManager] PlayerPrefs position unreadable: "
                               + $"'{PlayerPrefs.GetString(playerPrefsKeyPos)}'");
                return false;
            }
            if (!PrefsPose.TryParseVector3(PlayerPrefs.GetString(playerPrefsKeyRot), out var euler))
            {
                Debug.LogWarning("[SceneAnchorManager] PlayerPrefs rotation unreadable: "
                               + $"'{PlayerPrefs.GetString(playerPrefsKeyRot)}'");
                pos = Vector3.zero;
                return false;
            }
            rot = Quaternion.Euler(euler.x, euler.y, euler.z);
            return true;
        }
        catch (System.Exception ex)
        {
            Debug.LogWarning($"[SceneAnchorManager] PlayerPrefs pose parse failed: {ex.Message}");
            pos = Vector3.zero;
            rot = Quaternion.identity;
            return false;
        }
    }
}
