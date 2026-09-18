using UnityEngine;

/// <summary>
/// Intervention-mode controller for SIIScene_InterventionV1.
///
/// There is NO manual motion-controller toggle anymore (previously RIGHT A, then RIGHT
/// thumbstick click). Interventions are a single hard mode:
///
///   Left X  → intervention begins (Python aligns the robot, then publishes
///             "intervention_started" → InterventionStatusHud.InterventionActive = true)
///   While active:
///     - MotionControllerZmqPublisher is AUTO-ENABLED (streams right controller to mq3_mc.py;
///       harmless in telekinesis-only launches — nothing subscribes).
///     - All other inputs are blocked (Y/B/scene-exit) — see InterventionButtonForwarder /
///       InterventionBackButton. Only X (= accept/finish) and Left grip (= reject/cancel)
///       remain live.
///   Left X again (accept) or Left grip (reject) → Python publishes "intervention_finished"
///     → InterventionActive = false → publisher AUTO-DISABLED, controls restored.
///
/// This component just mirrors InterventionActive onto the publisher enable state and the
/// legacy MotionControllerModeActive static (still read by other components).
///
/// Call SetDependencies() from InterventionSessionBootstrap after attaching this component,
/// since Unity cannot serialise references to dynamically AddComponent'd objects.
/// </summary>
[DefaultExecutionOrder(-8900)]
public class MotionControllerModeManager : MonoBehaviour
{
    /// <summary>
    /// Mirrors InterventionStatusHud.InterventionActive (kept for existing readers).
    /// </summary>
    public static bool MotionControllerModeActive { get; private set; }

    /// <summary>
    /// True while the RIGHT controller belongs exclusively to the motion controller and
    /// must not be read as a UI input by anything else.
    ///
    /// This is the "MC control blocker". It was removed when scene-switch auto-cancel was
    /// added (the grid's select-on-trigger had to reach a live intervention to cancel it),
    /// but that traded away the property that matters more under MC: the right controller
    /// is the operator's HAND on the arm. A trigger pull meant as an MC clutch was landing
    /// on a grid panel behind the ray and switching sessions mid-takeover.
    ///
    /// Scoped to MC deliberately, so the auto-cancel-on-switch behaviour is retained in the
    /// KT and FACTR conditions, where the right controller drives nothing:
    ///   * McAdvertised  -- the launcher ran with MC_ACTIVE=1 (via the discovery beacon;
    ///                      the headset cannot read the launcher's env).
    ///   * intervention active -- Python published "intervention_started" and has not yet
    ///                      published "intervention_finished".
    /// The lock therefore ends exactly when the operator ACCEPTS (Left X) or REJECTS
    /// (Left grip) — both of which are LEFT-hand inputs and are never blocked.
    /// </summary>
    public static bool McControlsLocked { get; private set; }

    /// <summary>Whether this launcher is running the motion-controller condition, from the
    /// discovery beacon. Separated from McControlsLocked so the "am I in MC?" question and
    /// the "is the right hand busy right now?" question cannot drift apart.</summary>
    public static bool McAdvertised => PublisherDiscoveryListener.LastMcActive;

    private MotionControllerZmqPublisher _publisher;
    private InterventionStatusHud _hud;

    // ---------------------------------------------------------------- setup

    /// <summary>Called by InterventionSessionBootstrap to wire runtime refs.</summary>
    public void SetDependencies(MotionControllerZmqPublisher publisher, InterventionStatusHud hud)
    {
        _publisher = publisher;
        _hud = hud;
    }

    void Start()
    {
        MotionControllerModeActive = false;
        McControlsLocked = false;
        Debug.Log($"[MCModeManager] Ready — MC publisher auto-follows intervention state "
                + $"(X=accept, grip=reject). mcAdvertised={McAdvertised} "
                + $"(right-controller lock {(McAdvertised ? "WILL" : "will NOT")} engage during interventions).");
    }

    void OnDisable()
    {
        // Ensure mode is reset and publisher stops when the scene tears down. Releasing the
        // lock here matters: it is a static, so a scene unload that left it set would block
        // the right controller in the selector with no intervention anywhere to clear it.
        MotionControllerModeActive = false;
        McControlsLocked = false;
        if (_publisher != null) _publisher.enabled = false;
    }

    // ---------------------------------------------------------------- update

    void LateUpdate()
    {
        bool interventionActive = InterventionStatusHud.InterventionActive;

        // Recomputed every frame from the two inputs rather than latched, so it cannot be
        // left stuck on by a missed edge (e.g. an "intervention_finished" that arrives while
        // this component is disabled). Both inputs are cheap static reads.
        McControlsLocked = interventionActive && McAdvertised;

        if (interventionActive && !MotionControllerModeActive)
        {
            MotionControllerModeActive = true;
            if (_publisher != null) _publisher.enabled = true;
            _hud?.ShowWithColor(
                McAdvertised
                    ? "Intervention (MC) — right hand controls the arm · X: accept · Grip: reject"
                    : "Intervention — X: accept · Grip: reject",
                new Color(0f, 0.9f, 1f, 1f));
            Debug.Log($"[MCModeManager] Intervention began — MC publisher ON, "
                    + $"rightControllerLock={McControlsLocked}.");
        }
        else if (!interventionActive && MotionControllerModeActive)
        {
            MotionControllerModeActive = false;
            if (_publisher != null) _publisher.enabled = false;
            _hud?.ShowWithColor("Replay mode", Color.white, 2f);
            Debug.Log("[MCModeManager] Intervention ended — MC publisher OFF, right controller released.");
        }
    }
}
