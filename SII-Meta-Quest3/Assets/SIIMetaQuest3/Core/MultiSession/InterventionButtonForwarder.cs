using UnityEngine;

/// <summary>
/// In SIIScene_InterventionV1, forwards controller presses to the SELECTED
/// session's Python cmd_port (topic_port + 5), without depending on a working
/// XR-node connection (which is flaky across the multi-session setup).
///
/// Control scheme (Quest Touch: A/B = right controller, X/Y = left controller):
///   LEFT X            -> "INTERVENE" (starts intervention; during intervention = ACCEPT/finish)
///   LEFT grip trigger -> "CANCEL"    (during intervention = REJECT; cancels the replan)
///   RIGHT B           -> "SIM_TOGGLE" (sim start/stop — BLOCKED during intervention, and
///                                    skipped while the grid ray is on a panel, in which
///                                    case the selector pauses THAT panel instead).
///   LEFT Y            -> nothing. Reset is no longer exposed on the headset.
///   RIGHT A           -> empty (not forwarded; MC gripper button)
/// Scene enter/exit (RIGHT index trigger) is handled by InterventionBackButton (also blocked
/// during an intervention — the user must accept or reject first).
///
/// While PC-adjust mode is active (SceneAnchorManager.AdjustModeActive, toggled by
/// Left Menu), ALL forwarding is suppressed so the anchor-nudge controls (A/B/X/Y
/// rotation, sticks, L-triggers) don't fire sim/intervention/cancel.
///
/// Uses the same 1.5 s arming-delay + held-release-gate pattern as
/// InterventionBackButton so a button/trigger held during panel selection doesn't
/// fire a stray command on scene entry.
/// </summary>
[DefaultExecutionOrder(-9000)]
public class InterventionButtonForwarder : MonoBehaviour
{
    [Header("Arming")]
    public float armingDelaySeconds = 1.5f;
    public bool requireAllButtonsReleasedBeforeArming = true;

    private float _armingDeadline;
    // _armed is read only in the device build (LateUpdate is wrapped in
    // #if !UNITY_EDITOR), so the editor compile sees an assign-but-never-read
    // field. Silence the editor-only CS0414; the field is live on-device.
#pragma warning disable 0414
    private bool _armed;
#pragma warning restore 0414
    private string _ip;
    private int _cmdPort;

    void Start()
    {
        SessionRegistry.Load();
        if (!SessionRegistry.HasSelection)
        {
            enabled = false;
            return;
        }
        _ip = SessionRegistry.SelectedPublisherIp;
        _cmdPort = SessionRegistry.SelectedTopicPort + 5;
        _armingDeadline = Time.unscaledTime + Mathf.Max(0f, armingDelaySeconds);
        _armed = false;
        Debug.Log(
            $"[InterventionButtonForwarder] armed in {armingDelaySeconds:F1}s "
            + $"target=tcp://{_ip}:{_cmdPort} (X=intervene, B=sim, Lgrip=cancel)"
        );
    }

    /// <summary>Re-point forwarding to a different session in place (combined-view switch,
    /// no scene reload). Keeps B/Y/X/cancel targeting the correct session's cmd_port.</summary>
    public void Reconfigure(string ip, int cmdPort)
    {
        _ip = ip;
        _cmdPort = cmdPort;
        Debug.Log($"[InterventionButtonForwarder] reconfigured → target=tcp://{_ip}:{_cmdPort}");
    }

    void LateUpdate()
    {
#if !UNITY_EDITOR
        if (Time.unscaledTime < _armingDeadline) return;
        if (!_armed)
        {
            if (requireAllButtonsReleasedBeforeArming &&
                (OVRInput.Get(OVRInput.RawButton.X) ||
                 OVRInput.Get(OVRInput.RawButton.Y) ||
                 OVRInput.Get(OVRInput.RawButton.B) ||
                 OVRInput.Get(OVRInput.RawButton.LHandTrigger) ||
                 OVRInput.Get(OVRInput.RawButton.RIndexTrigger)))
                return;
            _armed = true;
            Debug.Log("[InterventionButtonForwarder] armed.");
        }

        // While adjusting the point-cloud anchor pose OR the risk chart placement, the face
        // buttons / sticks / triggers nudge that transform — do not forward them as
        // sim/intervention/cancel commands.
        if (SceneAnchorManager.AdjustModeActive || RiskBarChart.AdjustModeActive) return;

        // Intervention mode: only ACCEPT (X → INTERVENE toggles finish_replan) and
        // REJECT (Left grip → CANCEL) are live. B/Y are blocked — the user must accept or
        // reject the intervention to regain the other controls. mq3_mc.py no longer forwards
        // X/B/Y from the MC stream, so these are the single command path (no double-send).
        bool interventionMode = InterventionStatusHud.InterventionActive;

        if (OVRInput.GetDown(OVRInput.RawButton.X))
        {
            bool sent = SessionCommandSender.Send(_ip, _cmdPort, "INTERVENE");
            Debug.Log($"[InterventionButtonForwarder] X/{(interventionMode ? "ACCEPT" : "INTERVENE")} send result={sent}");
        }
        if (OVRInput.GetDown(OVRInput.RawButton.LHandTrigger))
        {
            bool sent = SessionCommandSender.Send(_ip, _cmdPort, "CANCEL");
            Debug.Log($"[InterventionButtonForwarder] Lgrip/{(interventionMode ? "REJECT" : "CANCEL")} send result={sent}");
        }

        if (interventionMode) return;   // B blocked during intervention

        // B = sim start/stop. Send the SEMANTIC name, not the raw letter: the bare letters
        // "B" and "Y" were interpreted differently by runtime_impl and app.py — app.py read
        // "B" as TASK_SUCCESS, so pressing pause saved the episode as a success and started
        // a new scene. Both sides still accept the letters as legacy aliases; nothing new
        // should ever send them.
        //
        // Combined view arbitration: the selector scene stays loaded, and its
        // MultiSessionGridManager also handles B. Whoever the ray is pointing at wins —
        // hovering a panel means "pause THAT scene" and the grid sends it; hovering
        // nothing means "pause the scene I am inside" and we send it. Without this gate a
        // single press fired at both, toggling two different sessions at once.
        // InterventionBackButton uses the same pattern for the exit trigger.
        if (OVRInput.GetDown(OVRInput.RawButton.B))
        {
            if (MultiSessionGridManager.CombinedViewEnabled
                && MultiSessionGridManager.IsHoveringSelectablePanel)
            {
                Debug.Log("[InterventionButtonForwarder] B ignored: the grid ray is on a panel, "
                          + "so the selector owns this press.");
            }
            else
            {
                bool sent = SessionCommandSender.Send(_ip, _cmdPort, "SIM_TOGGLE");
                Debug.Log($"[InterventionButtonForwarder] B/SIM_TOGGLE send result={sent}");
            }
        }
        // Y is deliberately NOT forwarded any more: reset is no longer exposed on the
        // headset. Python still accepts RESET/RESTART for the desktop 'R' key and
        // send_multi_window_command.py.
#endif
    }
}
