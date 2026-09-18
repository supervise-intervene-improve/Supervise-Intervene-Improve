using UnityEngine;

/// <summary>
/// Press-and-HOLD gate for the adjust/alignment mode toggle.
///
/// The toggle used to fire on the rising edge of Left Menu, so a single accidental brush of the
/// button unlocked the scene anchor mid-session — and because adjust mode suppresses ALL command
/// forwarding (intervention, sim, exit), the operator's next few button presses then silently did
/// nothing. Requiring a deliberate multi-second hold makes that misfire essentially impossible,
/// at the cost of nothing: unlocking is a rare, deliberate act.
///
/// Not a MonoBehaviour — it is a plain helper each toggle owner instantiates, so
/// SceneAnchorManager (single view) and MultiSessionGridManager (pure selector) share exactly one
/// implementation of the timing. They own the same physical button and must never disagree about
/// what counts as a hold.
///
/// Usage: call <see cref="Poll"/> once per frame with the button's CURRENT held state; it returns
/// true exactly once per hold, at the moment the threshold is crossed.
/// </summary>
public class AdjustModeHoldGate
{
    /// <summary>Default hold duration. Shared so both toggle sites start from the same value.</summary>
    public const float DefaultHoldSeconds = 3.0f;

    private float _pressStartedAt = -1f;
    private bool  _firedThisHold;
    private bool  _wasPressed;

    /// <summary>True on the frame the button went down — the moment to show "keep holding"
    /// feedback, since a 3 s hold with no response is indistinguishable from a dead button.</summary>
    public bool JustPressed { get; private set; }

    /// <summary>0→1 progress through the current hold. 0 when not pressed.</summary>
    public float Progress01 { get; private set; }

    /// <summary>
    /// Advance the gate. Returns true ONCE, on the frame the hold reaches holdSeconds.
    /// Releasing early cancels with no toggle; continuing to hold past the threshold does not
    /// re-fire (so one long press = one toggle, never a repeat).
    /// </summary>
    /// <param name="pressed">Is the button held THIS frame. Pass false to also cover
    /// "held, but a higher-priority mode owns the button right now" — that cancels the hold,
    /// which is the safe direction.</param>
    /// <param name="holdSeconds">Required hold duration. &lt;= 0 restores instant edge-trigger.</param>
    public bool Poll(bool pressed, float holdSeconds)
    {
        JustPressed = pressed && !_wasPressed;
        _wasPressed = pressed;

        if (!pressed)
        {
            _pressStartedAt = -1f;
            _firedThisHold  = false;
            Progress01      = 0f;
            return false;
        }

        // unscaledTime throughout: this must behave identically if anything ever pauses time.
        if (_pressStartedAt < 0f) _pressStartedAt = Time.unscaledTime;

        float required = Mathf.Max(0f, holdSeconds);
        float held     = Time.unscaledTime - _pressStartedAt;
        Progress01     = required <= 0f ? 1f : Mathf.Clamp01(held / required);

        if (_firedThisHold) return false;
        if (held < required) return false;

        _firedThisHold = true;
        Progress01     = 1f;
        return true;
    }

    /// <summary>Forget any in-progress hold. Call when another owner takes over the button, so a
    /// hold started under the old owner cannot complete under the new one.</summary>
    public void Reset()
    {
        _pressStartedAt = -1f;
        _firedThisHold  = false;
        _wasPressed     = false;
        JustPressed     = false;
        Progress01      = 0f;
    }
}
